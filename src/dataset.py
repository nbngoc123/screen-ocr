"""
dataset.py — PyTorch Dataset và DataLoader cho CRNN training.

Hỗ trợ 2 format data:
    - Synthetic: thư mục chứa labels.jsonl + ảnh .png
    - Real: thư mục chứa paired .png + .json files (từ crawler)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.augment import preprocess_for_crnn
from src.charset import CharsetCodec

logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """
    Dataset cho bài toán nhận diện text (text recognition).

    Đọc cả synthetic format (labels.jsonl) và real format (paired .json).
    Tự động shuffle samples khi khởi tạo.

    Args:
        data_dirs:     Danh sách thư mục data (có thể mix synthetic + real).
        charset_path:  Đường dẫn charset.txt.
        is_train:      True → dùng augmentation, False → chỉ normalize.
        max_label_len: Cắt label nếu dài hơn (để tránh CTC error).
        target_h:      Chiều cao ảnh sau resize (phải = 32 cho CRNN).

    Example:
        ds = OCRDataset(["data/synthetic/train", "data/real"], is_train=True)
        sample = ds[0]
        # sample["image"]     — tensor (1, 32, W)
        # sample["label"]     — tensor [int]
        # sample["label_len"] — tensor scalar
        # sample["label_str"] — str (raw text)
    """

    def __init__(
        self,
        data_dirs: list[str],
        charset_path: str = "data/charset.txt",
        is_train: bool = True,
        max_label_len: int = 80,
        target_h: int = 32,
    ) -> None:
        # TODO: implement — xem week1-data-pipeline.md §5.1
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: implement
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        # TODO: implement
        raise NotImplementedError


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: pad images về cùng width, pad labels về cùng length.

    Cần thiết vì CRNN nhận ảnh có width khác nhau trong cùng batch.

    Args:
        batch: List of dicts từ OCRDataset.__getitem__().

    Returns:
        Dict với tensors đã pad:
            image     — (B, 1, 32, max_W)
            label     — (B, max_L)
            label_len — (B,)
            input_len — (B,) chiều dài sequence sau CNN
            label_str — list[str]
    """
    # TODO: implement — xem week1-data-pipeline.md §5.1
    raise NotImplementedError


def get_dataloaders(
    train_dirs: list[str],
    val_dirs: list[str],
    charset_path: str = "data/charset.txt",
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Tạo train và val DataLoader.

    Args:
        train_dirs:   Thư mục training data.
        val_dirs:     Thư mục validation data.
        charset_path: Charset file.
        batch_size:   Batch size.
        num_workers:  Số worker process.

    Returns:
        (train_loader, val_loader)
    """
    # TODO: implement
    raise NotImplementedError
