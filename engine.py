from logging import exception
import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from PIL import Image
import numpy as np
import faster_coco_eval

# Replace pycocotools with faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EPS = 1e-16


def train_one_epoch(
    model,
    optimizer,
    train_loader,
    train_sampler,
    scheduler,
    epoch,
    epochs,
    rank,
    logger=None,
):
    # Epoch-level sampler shuffle
    train_sampler.set_epoch(epoch)

    # Training
    model.train()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    accu_loss = 0.0
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"[Train epoch {epoch}/{epochs}]", leave=False)
    else:
        pbar = train_loader

    for images, targets in pbar:
        images = [img.to(rank) for img in images]
        targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accu_loss += loss.item()
        if rank == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.3f}", lr=optimizer.param_groups[0]["lr"]
            )

            if logger is not None:
                logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

    scheduler.step()


def eval_one_epoch(
    model,
    val_loader,
    val_dataset,
    class_names,
    rank,
    epoch,
    epochs,
    det_metrics,
    logger,
):
    model.eval()
    stats = []
    pbar = tqdm(val_loader, desc=f"[Val epoch {epoch}/{epochs}]", leave=False)
    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(rank) for img in images]
            targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                pb, ps, pl = out["boxes"], out["scores"], out["labels"]
                tb, tl = tgt["boxes"], tgt["labels"]

                if pb.numel() and tb.numel():
                    iou = box_iou(pb, tb)
                    best_iou, best_idx = iou.max(dim=1)
                    match = pl == tl[best_idx]
                    iouv = np.linspace(0.5, 0.95, 10)
                    tp_matrix = np.zeros((iouv.shape[0], best_iou.shape[0]), dtype=bool)
                    best_iou_np = best_iou.cpu().numpy()
                    match_np = match.cpu().numpy()
                    for i, thr in enumerate(iouv):
                        tp_matrix[i] = (best_iou_np > thr) & match_np
                    tp = tp_matrix
                else:
                    num_det = pb.shape[0]
                    tp = np.zeros((10, num_det), dtype=bool)

                stats.append(
                    (
                        tp,
                        ps.cpu().numpy(),
                        pl.cpu().numpy(),
                        tl.cpu().numpy(),
                    )
                )

    tp_all, conf_all, pcls_all, tcls_all = zip(*stats)
    tp_all = np.concatenate(tp_all, axis=1)
    conf_all = np.concatenate(conf_all, axis=0)
    pcls_all = np.concatenate(pcls_all, axis=0)
    tcls_all = np.concatenate(tcls_all, axis=0)

    det_metrics.process(
        tp=tp_all,
        conf=conf_all,
        pred_cls=pcls_all,
        target_cls=tcls_all,
    )
    results = det_metrics.results_dict

    mAP5095 = results["metrics/mAP50-95(B)"]
    mAP50 = results["metrics/mAP50(B)"]
    ar = results["metrics/recall(B)"]
    ap = results["metrics/precision(B)"]

    stats = {
        "eval/mAP5095": mAP5095,
        "eval/mAP50": mAP50,
        "eval/recall": ar,
        "eval/precision": ap,
    }

    if logger is not None:
        logger.log(stats)
    return stats


# def eval_one_epoch(
#     model,
#     val_loader,
#     val_dataset,
#     class_names,
#     rank,
#     epoch,
#     epochs,
#     logger,
# ):
#     model.eval()
#     all_dets, all_targets = [], []
#     pbar = tqdm(val_loader, desc=f"[Val epoch {epoch}/{epochs}]", leave=False)
#     with torch.no_grad():
#         for images, targets in pbar:
#             images = [img.to(rank) for img in images]
#             targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]
#
#             try:
#                 outputs = model(images)
#             except Exception as e:
#                 print("images:", images)
#                 print("targets:", targets)
#                 print("Exception during model inference:", e)
#
#             for out in outputs:
#                 all_dets.append(
#                     {
#                         "boxes": out["boxes"].cpu(),
#                         "scores": out["scores"].cpu(),
#                         "labels": out["labels"].cpu(),
#                     }
#                 )
#
#             for t in targets:
#                 all_targets.append(
#                     {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
#                 )
#
#     coco_gt = {
#         "images": [],
#         "annotations": [],
#         "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
#     }
#     ann_id = 0
#     for img_id, (tgt, img_path) in enumerate(zip(all_targets, val_dataset.images)):
#         w, h = Image.open(img_path).size
#         coco_gt["images"].append(
#             {"id": img_id, "width": w, "height": h, "file_name": img_path.name}
#         )
#         for box, lbl in zip(tgt["boxes"], tgt["labels"]):
#             x1, y1, x2, y2 = box.tolist()
#             ww, hh = x2 - x1, y2 - y1
#             coco_gt["annotations"].append(
#                 {
#                     "id": ann_id,
#                     "image_id": img_id,
#                     "category_id": int(lbl.item()),
#                     "bbox": [x1, y1, ww, hh],
#                     "area": ww * hh,
#                     "iscrowd": 0,
#                 }
#             )
#             ann_id += 1
#
#     cocoGt = COCO()
#     cocoGt.dataset = coco_gt
#     cocoGt.createIndex()
#
#     coco_dt = []
#     for img_id, det in enumerate(all_dets):
#         for box, score, lbl in zip(det["boxes"], det["scores"], det["labels"]):
#             x1, y1, x2, y2 = box.tolist()
#             ww, hh = x2 - x1, y2 - y1
#             coco_dt.append(
#                 {
#                     "image_id": img_id,
#                     "category_id": int(lbl.item()),
#                     "bbox": [x1, y1, ww, hh],
#                     "score": float(score.item()),
#                 }
#             )
#     if len(coco_dt) == 0:
#         print("Warning: no detections in this epoch; skipping COCO eval.")
#         return {
#             "eval/mAP5095": 0.0,
#             "eval/mAP50": 0.0,
#             "eval/recall": 0.0,
#             "eval/precision": 0.0,
#         }
#     cocoDt = cocoGt.loadRes(coco_dt)
#     cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#
#     stats = cocoEval.stats
#     mAP5095 = float(stats[0])
#     mAP50 = float(stats[1])
#
#     recall_mat = cocoEval.eval["recall"]
#     # recall   : [T, K, A, M]      (IoU thresh, class, area, maxDet)
#     recall50 = recall_mat[0, :, 0, 2]
#     avg_recall = float(np.nanmean(recall50))
#
#     prec_mat = cocoEval.eval["precision"]
#     # precision: [T, R, K, A, M]   (IoU thresh, recall thresh, class, area, maxDet)
#     precision50 = prec_mat[0, :, :, 0, 2]
#     avg_prec = float(np.nanmean(precision50))
#
#     print(f"[DEBUG PRINT] avg_recall: {avg_recall}, avg_prec: {avg_prec}")
#
#     stats = {
#         "eval/mAP5095": mAP5095,
#         "eval/mAP50": mAP50,
#         "eval/recall": avg_recall,
#         "eval/precision": avg_prec,
#     }
#
#     if logger is not None:
#         logger.log(stats)
#     return stats
