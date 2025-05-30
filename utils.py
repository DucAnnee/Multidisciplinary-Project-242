import os
import wandb
from PIL import ImageDraw, ImageFont


def wandb_init(lr0, epochs, batch_size, project="mp242"):
    api_key = os.getenv("WANDB_API_KEY", None)

    if api_key is None:
        try:
            from kaggle_secrets import UserSecretsClient

            api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            api_key = None

    # Skip if no api_key
    if not api_key:
        print("WANDB_API_KEY not found; skipping wandb.init()")
        return None

    # os.environ["WANDB_API_KEY"] = api_key
    wandb.login(key=api_key)

    logger = wandb.init(
        project=project,
        config={"lr0": lr0, "epochs": epochs, "batch_size": batch_size},
    )
    logger.define_metric("eval/precision", summary="max")
    logger.define_metric("eval/recall", summary="max")
    logger.define_metric("eval/mAP50", summary="max")
    logger.define_metric("eval/mAP5095", summary="max")

    return logger


def draw_detections(
    image,
    outputs,
    class_names,
    score_thresh=0.5,
    box_color="red",
    text_color="white",
    font=None,
):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    if font is None:
        font = ImageFont.load_default()

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    for (x1, y1, x2, y2), lab, scr in zip(boxes, labels, scores):
        if scr < score_thresh:
            continue

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        text = f"{class_names[lab]}: {scr:.2f}"

        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            mask = font.getmask(text)
            text_w, text_h = mask.size

        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=box_color)
        draw.text((x1, y1 - text_h), text, fill=text_color, font=font)

    return img
