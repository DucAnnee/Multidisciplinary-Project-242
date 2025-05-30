import argparse
import os
import torch
from pathlib import Path
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from model import RetinaClassificationHeadDropout
from torchvision.transforms import functional as F
from PIL import Image
from utils import draw_detections


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with RetinaNet model and draw bounding boxes."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="best.pth",
        help="Path to the trained checkpoint file",
    )
    parser.add_argument(
        "--input-img",
        type=str,
        required=True,
        help="Path to the input image for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./out_imgs",
        help="Directory to save output image (default: ./out_imgs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for drawing boxes (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # load image
    img_path = Path(args.input_img)
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img)

    # prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    num_classes = 2
    dropout_p = 0.1
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaClassificationHeadDropout(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        dropout=dropout_p,
    )

    # load weights
    weights = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        model.load_state_dict(weights["model_state_dict"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # inference
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model([img_tensor])[0]

    # draw and save
    vis = draw_detections(
        img, outputs, ["diseased", "healthy"], score_thresh=args.threshold
    )
    out_name = img_path.stem + "_out" + img_path.suffix
    vis.save(os.path.join(args.output_dir, out_name))
    print(f"Inference result saved to {os.path.join(args.output_dir, out_name)}")


if __name__ == "__main__":
    main()
