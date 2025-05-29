import random
import yaml

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
    # Hyperparameters and paths
    data_yaml = args.data_yaml
    project_dir = args.project_dir
    train_dir = args.train_dir
    val_dir = args.val_dir
    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    lr0 = args.lr0
    momentum = args.momentum
    weight_decay = args.weight_decay
    dropout_p = args.dropout_p
    num_classes = args.num_classes
    seed = args.seed
    enable_logger = args.enable_logger

    # Output dirs
    (project_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Load data.yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    class_names = data["names"]

    # DDP init
    device = ddp_init(rank, world_size)

    # Seed set
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    # Dataset, DistributedSampler, DataLoader
    train_dataset, train_loader, train_sampler = build_dataloader(
        train_dir, img_size, rank, world_size, True, batch_size
    )
    val_dataset, val_loader, val_sampler = build_dataloader(
        val_dir, img_size, rank, world_size, False, batch_size
    )

    # Model
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    num_anchors = model.head.classification_head.num_anchors
    # replace head
    model.head.classification_head = RetinaClassificationHeadDropout(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        dropout=dropout_p,
    )
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Optimizer, LR scheduler
    optimizer = optim.SGD(
        model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # WandB
    logger = None
    if rank == 0 and enable_logger:
        logger = wandb_init(lr0, epochs, batch_size)

    # Training loop
    best_map = -1.0

    for epoch in range(1, epochs + 1):
        # Train
        train_one_epoch(
            model,
            optimizer,
            train_loader,
            train_sampler,
            scheduler,
            epoch,
            epochs,
            rank,
            logger,
        )

        # Validation
        if rank == 0:
            stats = eval_one_epoch(
                model, val_loader, val_dataset, class_names, rank, logger
            )
            mAP5095 = stats["eval/mAP5095"]

            if mAP5095 > best_map:
                best_map = mAP5095
                torch.save(
                    model.module.state_dict(), project_dir / "checkpoints" / "best.pth"
                )

            ckpt = {
                "epoch": epoch,
                "best_map": best_map,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }

            ckpt_path = project_dir / "checkpoints" / f"retina_ep{epoch:03d}.pth"
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
        "--data_yaml",
        type=str,
        default="/kaggle/input/bok-choy-disease-detection-yolo-format/data.yaml",
        help="Path to the data.yaml file",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="/kaggle/working/retinanet_bokchoy/",
        help="Directory for outputs (checkpoints, logs, etc.)",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/kaggle/input/bok-choy-disease-detection-yolo-format/train",
        help="Training images directory",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/kaggle/input/bok-choy-disease-detection-yolo-format/valid",
        help="Validation images directory",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--img_size", type=int, default=640, help="Input image size (square)"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.1,
        help="Dropout probability in classification head",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of target classes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--enable-logger",
        type=int,
        default=0,
        help="Enable WandB logging. Defaults to 0 (disabled) ",
    )

    args = parser.parse_args()

    args.project_dir = Path(args.project_dir)
    mp.set_start_method("fork", force=True)
    main(args)
