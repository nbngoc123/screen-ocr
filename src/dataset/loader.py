import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.dataset.charset import CharsetCodec
from src.data_generation import preprocess_for_crnn

class OCRDataset(Dataset):
    def __init__(
        self,
        data_dirs: list[str],
        charset_path: str = "data/charset.txt",
        is_train: bool = True,
        max_label_len: int = 80,
        target_h: int = 48,
    ):
        self.codec = CharsetCodec(charset_path)
        self.is_train = is_train
        self.max_label_len = max_label_len
        self.target_h = target_h
        self.samples = self._load_samples(data_dirs)
        print(f"Dataset: {len(self.samples)} samples, train={is_train}")

    def _load_samples(self, data_dirs: list[str]) -> list[dict]:
        samples = []
        for d in data_dirs:
            p = Path(d)
            if not p.exists():
                continue
                
            labels_file = p / "labels.jsonl"
            if labels_file.exists():
                # Format dùng chung cho Synthetic và ICDAR crop
                with open(labels_file, encoding="utf-8") as f:
                    for line in f:
                        m = json.loads(line)
                        img_path = p / m["file"]
                        if img_path.exists():
                            samples.append({"img": str(img_path), "label": m["label"]})
            else:
                # Tìm tất cả file .json nếu không có labels.jsonl
                for img_path in sorted(p.glob("*.png")):
                    json_path = img_path.with_suffix(".json")
                    if json_path.exists():
                        meta = json.loads(json_path.read_text(encoding="utf-8"))
                        samples.append({"img": str(img_path), "label": meta["label"]})

        random.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img = cv2.imread(sample["img"])
        if img is None:
            # Fallback: blank image nếu lỗi
            img = np.ones((self.target_h, 100, 3), dtype=np.uint8) * 255

        # Preprocess
        img_tensor = preprocess_for_crnn(img, self.target_h, self.is_train)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()  # (1, H, W)

        # Encode label
        label = sample["label"][:self.max_label_len]
        encoded = self.codec.encode(label)

        return {
            "image":        img_tensor,
            "label":        torch.tensor(encoded, dtype=torch.long),
            "label_len":    torch.tensor(len(encoded), dtype=torch.long),
            "label_str":    label,
            "img_path":     sample["img"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad images về cùng width, pad labels về cùng length."""
    # Pad images (chiều rộng khác nhau)
    max_w = max(b["image"].shape[2] for b in batch)
    h = batch[0]["image"].shape[1]
    images = torch.zeros(len(batch), 1, h, max_w)
    for i, b in enumerate(batch):
        w = b["image"].shape[2]
        images[i, :, :, :w] = b["image"]

    # Pad labels
    max_l = max(len(b["label"]) for b in batch)
    labels = torch.zeros(len(batch), max_l, dtype=torch.long)
    for i, b in enumerate(batch):
        l = len(b["label"])
        labels[i, :l] = b["label"]

    return {
        "image":       images,
        "label":       labels,
        "label_len":   torch.stack([b["label_len"] for b in batch]),
        "label_str":   [b["label_str"] for b in batch],
        "input_len":   torch.tensor(
            [max_w // 4 for _ in batch], dtype=torch.long  # output width sau CNN layer (stride = 4)
        ),
    }


def get_dataloaders(
    train_dirs: list[str],
    val_dirs: list[str],
    charset_path: str = "data/charset.txt",
    batch_size: int = 256,
    num_workers: int = 4,
    target_h: int = 48,
) -> tuple[DataLoader, DataLoader]:

    train_ds = OCRDataset(train_dirs, charset_path, is_train=True, target_h=target_h)
    val_ds   = OCRDataset(val_dirs,   charset_path, is_train=False, target_h=target_h)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
