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
import albumentations as A

logger = logging.getLogger(__name__)

# ─── Training transform (moderate augmentation) ───────────────────────────────
# Pipeline cho training — moderate augmentation
# Lưu ý: Không dùng Rotation mạnh vì text trên màn hình đa số nằm ngang hoàn hảo
TRAIN_TRANSFORM = A.Compose([
    A.GaussianBlur(blur_limit=(1, 3), p=0.25),
    A.ImageCompression(quality_range=(70, 95), p=0.2),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.3,
    ),
    A.GaussNoise(p=0.2),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.15),
    # Normalize về [-1, 1]
    A.Normalize(mean=(0.5,), std=(0.5,)),
])

# ─── Val/Test transform (normalize only) ─────────────────────────────────────
# Pipeline cho val/test — chỉ normalize
VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
])


def preprocess_for_crnn(
    image: np.ndarray,
    target_h: int = 48,
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
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image size: {w}×{h}")

    # Resize giữ nguyên tỉ lệ (Aspect Ratio)
    new_w = max(int(w * target_h / h), 1)
    image = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Augment (yêu cầu image có 3 chiều cho albumentations, ta phải expand_dims)
    transform = TRAIN_TRANSFORM if is_train else VAL_TRANSFORM
    augmented = transform(image=image[..., np.newaxis])["image"]

    # Đưa về lại (H, W) float32
    return augmented.squeeze(-1)
