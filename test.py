from argparse import ArgumentParser
from pathlib import Path
import yaml

import torch
import torch.multiprocessing as mp

from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from engine import eval_one_epoch
from data import build_dataloader
from model import RetinaClassificationHeadDropout


def main(args):
    # Load data.yaml
    class_names = None
    if Path(args.data_yaml).exists():
        with open(args.data_yaml) as f:
            data = yaml.safe_load(f)
        class_names = data["names"]
        test_dir = data.get("test", None)

    # Model
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    num_anchors = model.head.classification_head.num_anchors
    # replace head
    model.head.classification_head = RetinaClassificationHeadDropout(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=args.num_classes,
        dropout=args.dropout_p,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load weights
    weights = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        model.load_state_dict(weights["model_state_dict"])
    else:
        model.load_state_dict(weights)
    model.to(device)

    test_dataset, test_loader, test_sampler = build_dataloader(
        test_dir,
        args.dataset_format,
        args.img_size,
        0,
        1,
        False,
        args.batch_size,
        None,
    )

    stats = eval_one_epoch(
        model,
        test_loader,
        test_dataset,
        class_names,
        0,
        1,
        args.epochs,
        None,
    )

    print(stats)


if __name__ == "__main__":
    parser = ArgumentParser(description="MP242 Training Script")
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="/kaggle/input/bok-choy-disease-detection-yolo-format/data.yaml",
        help="Path to the data.yaml file",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--img-size", type=int, default=640, help="Input image size (square)"
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

    parser.add_argument(
        "--dataset-format",
        type=str,
        default="yolo",
        choices=["yolo", "coco"],
        help="Format of the dataset",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="best.pth",
        help="Path to the trained checkpoint file",
    )

    args = parser.parse_args()

    mp.set_start_method("fork", force=True)
    main(args)
