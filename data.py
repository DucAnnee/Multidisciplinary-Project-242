from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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


class YoloFormatDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.transforms = transforms

        img_dir = Path(img_dir)
        if not img_dir.exists():
            raise ValueError(f"Path not found: {img_dir}")

        if img_dir.is_dir() and (img_dir / "images").is_dir():
            images_path = img_dir / "images"
        else:
            images_path = img_dir

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
        self.images = []
        for ext in exts:
            self.images += list(images_path.glob(ext))
            self.images += list(images_path.glob(ext.upper()))

        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_path!r}")

        self.labels_dir = images_path.parent / "labels"
        if not self.labels_dir.is_dir():
            raise ValueError(f"Labels folder not found: {self.labels_dir!r}")

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
                try:
                    c, xc, yc, w, h = tuple(map(float, line.strip().split()[:5]))
                except:
                    print(line)
                    raise ValueError()
                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c))

        if self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target


def build_dataloader(
    data_dir,
    img_size,
    rank,
    world_size,
    shuffle,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    dataset = YoloFormatDataset(data_dir, transforms=get_transform(img_size))
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
