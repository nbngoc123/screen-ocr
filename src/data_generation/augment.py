"""
augment.py — Albumentations preprocessing pipeline cho CRNN.
"""
from __future__ import annotations

import logging
import yaml
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

logger = logging.getLogger(__name__)

# Load config
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"

try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    _aug_cfg = _config.get("augmentation", {})
except Exception as e:
    logger.warning(f"Không thể load cấu hình augmentation từ {_CONFIG_PATH}: {e}. Sẽ dùng giá trị mặc định.")
    _aug_cfg = {}

# Đọc các giá trị config hoặc lấy mặc định nếu không có
BLUR_LIMIT = (_aug_cfg.get("blur_limit_min", 1), _aug_cfg.get("blur_limit_max", 3))
BLUR_PROB = _aug_cfg.get("blur_prob", 0.25)

COMP_QUAL = (_aug_cfg.get("compression_quality_min", 70), _aug_cfg.get("compression_quality_max", 95))
COMP_PROB = _aug_cfg.get("compression_prob", 0.2)

BRIGHTNESS_LIMIT = _aug_cfg.get("brightness_limit", 0.15)
CONTRAST_LIMIT = _aug_cfg.get("contrast_limit", 0.15)
BRIGHT_CONT_PROB = _aug_cfg.get("brightness_contrast_prob", 0.3)

NOISE_PROB = _aug_cfg.get("noise_prob", 0.2)

SHARPEN_ALPHA = (_aug_cfg.get("sharpen_alpha_min", 0.1), _aug_cfg.get("sharpen_alpha_max", 0.3))
SHARPEN_LIGHT = (_aug_cfg.get("sharpen_lightness_min", 0.8), _aug_cfg.get("sharpen_lightness_max", 1.0))
SHARPEN_PROB = _aug_cfg.get("sharpen_prob", 0.15)

# ─── Training transform (moderate augmentation) ───────────────────────────────
# Pipeline cho training — moderate augmentation
# Lưu ý: Không dùng Rotation mạnh vì text trên màn hình đa số nằm ngang hoàn hảo
TRAIN_TRANSFORM = A.Compose([
    A.GaussianBlur(blur_limit=BLUR_LIMIT, p=BLUR_PROB),
    A.ImageCompression(quality_range=COMP_QUAL, p=COMP_PROB),
    A.RandomBrightnessContrast(
        brightness_limit=BRIGHTNESS_LIMIT,
        contrast_limit=CONTRAST_LIMIT,
        p=BRIGHT_CONT_PROB,
    ),
    A.GaussNoise(p=NOISE_PROB),
    A.Sharpen(alpha=SHARPEN_ALPHA, lightness=SHARPEN_LIGHT, p=SHARPEN_PROB),
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
    target_h: int = 64,
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
        target_h: Chiều cao target (CRNN sẽ nhận H này).
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
