import random
import yaml
import json

from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import wandb_init
from engine import train_one_epoch, eval_one_epoch
from ddp import ddp_init
from data import build_dataloader
from model import RetinaClassificationHeadDropout


def main_worker(rank, world_size, args):
    train_dir = args.train_dir
    val_dir = args.val_dir

    # Output dirs
    (args.project_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.project_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.project_dir / "eval").mkdir(parents=True, exist_ok=True)

    # Load data.yaml
    class_names = None
    if Path(args.data_yaml).exists():
        with open(args.data_yaml) as f:
            data = yaml.safe_load(f)
        class_names = data["names"]
        train_dir = data.get("train", None)
        # test_dir = data.get("test", None)
        val_dir = data.get("val", None)

    # DDP init
    device = ddp_init(rank, world_size)

    # Seed set
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # Dataset, DistributedSampler, DataLoader
    train_dataset, train_loader, train_sampler = build_dataloader(
        train_dir,
        args.dataset_format,
        args.img_size,
        rank,
        world_size,
        True,
        args.batch_size,
        args.train_coco_json,
    )
    val_dataset, val_loader, val_sampler = build_dataloader(
        val_dir,
        args.dataset_format,
        args.img_size,
        rank,
        world_size,
        False,
        args.batch_size,
        args.test_coco_json,
    )

    # Model
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    # Freeze all backbone (ResNet50+FPN) parameters
    for _, param in model.backbone.named_parameters():
        param.requires_grad = False
    for _, param in model.backbone.body.named_parameters():
        param.requires_grad = False
    for _, param in model.backbone.fpn.named_parameters():
        param.requires_grad = False
    # replace head
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaClassificationHeadDropout(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=args.num_classes,
        dropout=args.dropout_p,
    )
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Optimizer, LR scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lrf
    )

    # WandB
    logger = None
    if rank == 0 and args.enable_logger:
        logger = wandb_init(
            args.lr0,
            args.epochs,
            args.batch_size,
            args.wandb_entity,
            args.wandb_project,
        )

    # Training loop
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_one_epoch(
            model,
            optimizer,
            train_loader,
            train_sampler,
            scheduler,
            epoch,
            args.epochs,
            rank,
            logger,
        )

        # Validation
        if rank == 0:
            stats = eval_one_epoch(
                model,
                val_loader,
                val_dataset,
                class_names,
                rank,
                epoch,
                args.epochs,
                logger,
            )
            mAP5095 = stats["eval/mAP5095"]

            if mAP5095 > best_map:
                best_map = mAP5095
                torch.save(
                    model.module.state_dict(),
                    args.project_dir / "checkpoints" / "best.pth",
                )
                with open(args.project_dir / "eval" / "best_eval.txt", "w") as f:
                    json.dump(stats, f, indent=2)

            ckpt = {
                "epoch": epoch,
                "best_map": best_map,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }

            ckpt_path = args.project_dir / "checkpoints" / f"last.pth"
            torch.save(ckpt, ckpt_path)

    if rank == 0:
        logger.finish()
    dist.destroy_process_group()


def main(args):
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        args=(
            world_size,
            args,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="MP242 Training Script")
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="/kaggle/input/bok-choy-disease-detection-yolo-format/data.yaml",
        help="Path to the data.yaml file",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default="/kaggle/working/retinanet_bokchoy/",
        help="Directory for outputs (checkpoints, logs, etc.)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/kaggle/input/bok-choy-disease-detection-yolo-format/train",
        help="Training images directory",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="/kaggle/input/bok-choy-disease-detection-yolo-format/valid",
        help="Validation images directory",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--img-size", type=int, default=640, help="Input image size (square)"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.010, help="Initial learning rate"
    )
    parser.add_argument(
        "--lrf", type=float, default=0.00001, help="Final learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.1,
        help="Dropout probability in classification head",
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Number of target classes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--enable-logger",
        type=int,
        default=0,
        help="Enable WandB logging. Defaults to 0 (disabled) ",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mp242",
        help="Name of the WandB project",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Entity of the WandB to save the log to",
    )

    parser.add_argument(
        "--dataset-format",
        type=str,
        default="yolo",
        choices=["yolo", "coco"],
        help="Format of the dataset",
    )
    parser.add_argument(
        "--train-coco-json",
        type=str,
        default=None,
        help="Path to the json of the COCO-format train dataset",
    )
    parser.add_argument(
        "--test-coco-json",
        type=str,
        default=None,
        help="Path to the json of the COCO-format test dataset",
    )

    args = parser.parse_args()

    args.project_dir = Path(args.project_dir)
    mp.set_start_method("fork", force=True)
    main(args)
