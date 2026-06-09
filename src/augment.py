"""
augment.py — Albumentations preprocessing pipeline cho CRNN.

Quy tắc augmentation (screen-ocr-rules.md §2.4):
    ĐƯỢC PHÉP: gaussian_blur, jpeg_compression, brightness_contrast,
               gaussian_noise, slight_shear (max ±3°)
    CẤM:       rotation > 5°, perspective_transform, elastic_transform

    Lý do: Screen text luôn nằm ngang, không bị distort.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Training transform (moderate augmentation) ───────────────────────────────
# TODO: khởi tạo A.Compose([...]) theo screen-ocr-project.md §Data Pipeline
# Xem week1-data-pipeline.md §3.1 để biết chi tiết các transform
TRAIN_TRANSFORM = None  # placeholder — thay bằng A.Compose([...])

# ─── Val/Test transform (normalize only) ─────────────────────────────────────
VAL_TRANSFORM = None  # placeholder — thay bằng A.Compose([A.Normalize(...)])


def preprocess_for_crnn(
    image: np.ndarray,
    target_h: int = 32,
    is_train: bool = True,
) -> np.ndarray:
    """
    Chuẩn bị ảnh cho CRNN input.

    Pipeline:
        1. Validate input (không None, không quá nhỏ)
        2. Resize về H=target_h, giữ aspect ratio
        3. Convert grayscale (CRNN input là 1-channel)
        4. Augment (train) hoặc chỉ normalize (val/test)

    Args:
        image:    BGR uint8 numpy array.
        target_h: Chiều cao target (CRNN cần H=32).
        is_train: True → dùng TRAIN_TRANSFORM, False → VAL_TRANSFORM.

    Returns:
        float32 array shape (H, W), normalize về [-1, 1].

    Raises:
        ValueError: Nếu image có size không hợp lệ.
    """
    # TODO: implement — xem week1-data-pipeline.md §3.1
    raise NotImplementedError
