from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pycocotools.coco import COCO
from PIL import Image
from torchvision import transforms


# Transform
def get_transform(img_size):
    if img_size is None:
        img_size = 640

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return transform


def collate_fn(batch):
    return tuple(zip(*batch))


class COCODataset(Dataset):
    def __init__(
        self,
        coco_json,
        img_dir,
        transforms,
    ):
        super().__init__()
        self.coco = COCO(str(coco_json))
        self.img_dir = Path(img_dir)
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        coco_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(coco_id)[0]
        file_name = img_info["file_name"]
        width, height = img_info["width"], img_info["height"]

        img_path = self.img_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=coco_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes: list[list[float]] = []
        labels: list[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            # convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        if self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([coco_id]),
        }
        return img, target


class YoloDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.transforms = transforms

        folder_type = img_dir.split("/")[-1]
        img_dir = Path(img_dir)
        if not img_dir.exists():
            raise ValueError(f"Path not found: {img_dir}")

        if img_dir.is_dir() and (img_dir / "images").is_dir():
            images_path = img_dir / "images"
        else:
            images_path = img_dir

        if (images_path.parent / "labels").is_dir():
            self.labels_dir = images_path.parent / "labels"
        else:
            self.labels_dir = images_path.parent.parent / "labels" / folder_type

        if not self.labels_dir.is_dir():
            raise ValueError(f"Labels folder not found: {self.labels_dir!r}")

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
        self.images = []
        for ext in exts:
            self.images += list(images_path.glob(ext))
            self.images += list(images_path.glob(ext.upper()))

        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_path!r}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        stem = img_path.stem
        label_path = self.labels_dir / f"{stem}.txt"

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        boxes, labels = [], []
        if label_path.exists():
            for line in open(label_path):
                if len(line.strip().split()) > 5:
                    print(f"Skipping invalid line in {label_path}: {line.strip()}")
                    continue
                try:
                    c, xc, yc, w, h = tuple(map(float, line.strip().split()))
                except:
                    raise ValueError("Invalid label format in line: " + line.strip())
                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c))
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.empty((0, 4), dtype=torch.float32)
            labels_t = torch.empty((0,), dtype=torch.int64)
        if self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
        }
        return img, target


def build_dataloader(
    data_dir,
    dataset_format,
    img_size,
    rank,
    world_size,
    shuffle,
    batch_size,
    coco_json=None,
    num_workers=4,
    pin_memory=True,
):
    if dataset_format == "yolo":
        dataset = YoloDataset(data_dir, transforms=get_transform(img_size))
    elif dataset_format == "coco":
        assert (
            coco_json is not None
        ), "Please provide path to json of the COCO-format dataset"
        dataset = COCODataset(coco_json, data_dir, get_transform(img_size))
    else:
        raise ValueError("dataset_format is invalid, choices are 'yolo' or 'coco'")

    data_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=data_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return dataset, data_loader, data_sampler
