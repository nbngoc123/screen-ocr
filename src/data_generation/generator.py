"""
data_gen.py — Synthetic image generator.

Sinh ảnh chứa text từ Windows fonts với nhiều màu nền/chữ khác nhau.
Ground truth 100% chính xác vì text được render từ code.

Quy tắc (screen-ocr-rules.md §6.3):
    - Cache font objects bằng @lru_cache — không load font trong vòng lặp.
    - Augmentation chỉ dùng blur/noise/brightness — không rotation/perspective.
"""
from __future__ import annotations

import json
import logging
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.corpus import random_text

logger = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Load và cache font object.
    
    Lưu ý: Pillow không tự fallback sang font khác khi thiếu glyph. 
    Nếu kí tự không được hỗ trợ (OOV glyph), nó sẽ hiển thị dạng hộp vuông (tofu).
    """
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _rand_color(lo: int, hi: int) -> tuple[int, int, int]:
    """Sinh màu RGB ngẫu nhiên trong range [lo, hi]."""
    return tuple(random.randint(lo, hi) for _ in range(3))


def _ensure_contrast(
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    min_diff: int = 80,
) -> tuple[int, int, int]:
    """Đảm bảo foreground đủ contrast so với background."""
    diff = sum(abs(b - f) for b, f in zip(bg, fg)) / 3
    if diff < min_diff:
        # Đảo màu foreground
        fg = tuple(255 - c for c in fg)
    return fg


@lru_cache(maxsize=512)
def _get_tofu_bbox(font_path: str, size: int):
    """Lấy bounding box của ký tự không tồn tại (sẽ sinh ra tofu/ô vuông)."""
    font = _load_font(font_path, size)
    return font.getbbox("\ufffe")


@lru_cache(maxsize=8192)
def _is_char_supported(char: str, font_path: str, size: int) -> bool:
    """Kiểm tra một ký tự có được support bởi font hay bị biến thành tofu."""
    if char.isspace():
        return True
    font = _load_font(font_path, size)
    tofu_bbox = _get_tofu_bbox(font_path, size)
    char_bbox = font.getbbox(char)
    
    # Nếu bbox của ký tự giống hệt bbox của tofu, tức là font không support
    # (Đa số các trường hợp tofu có một kích thước hộp vuông cố định)
    return char_bbox != tofu_bbox


def check_text_fits_font(text: str, font_path: str, size: int) -> bool:
    """Kiểm tra xem font có chứa đủ tất cả glyph cho text không."""
    for c in text:
        if not _is_char_supported(c, font_path, size):
            return False
    return True


def generate_sample(
    text: str,
    font_path: str,
    font_size: int | None = None,
    padding: int = 8,
) -> np.ndarray | None:
    """
    Sinh một ảnh chứa text.
    """
    size = font_size or random.randint(10, 52)
    
    # KIỂM TRA FONT: Nếu font không hỗ trợ chữ này -> Bỏ qua ngay lập tức
    if not check_text_fits_font(text, font_path, size):
        return None
        
    font = _load_font(font_path, size)

    try:
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0] + padding * 2
        h = bbox[3] - bbox[1] + padding * 2
        if w < 4 or h < 4:
            return None
    except Exception:
        return None

    # Sinh background ngẫu nhiên
    bg_mode = random.choice(["solid", "light_gradient_sim", "near_white"])
    if bg_mode == "solid":
        bg = _rand_color(180, 255)
    elif bg_mode == "near_white":
        bg = _rand_color(230, 255)
    else:
        bg = _rand_color(200, 245)

    # Sinh foreground ngẫu nhiên
    fg = _rand_color(0, 80) if random.random() < 0.8 else _rand_color(150, 255)
    fg = _ensure_contrast(bg, fg)

    # Render ảnh
    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding - bbox[1]), text, font=font, fill=fg)

    # Convert RGB to BGR cho OpenCV
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_dataset(
    font_list: list[str],
    output_dir: str,
    n_samples: int = 500_000,
    font_size_range: tuple[int, int] = (10, 52),
    font_weights: list[float] | None = None,
    worker_id: int = 0,
    corpus_prob: float = 0.8,
    word_split_prob: float = 0.3,
    phrase_split_prob: float = 0.4,
) -> None:
    """Sinh toàn bộ synthetic dataset ra thư mục."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    count = 0

    # Mở file ghi liên tục (append hoặc write) để không bị mất dữ liệu nếu user bấm Ctrl+C
    worker_label_file = out / f"labels_{worker_id}.jsonl"
    with open(worker_label_file, "w", encoding="utf-8") as f_json:
        with tqdm(total=n_samples, desc=f"Worker {worker_id}", unit="img", position=worker_id) as pbar:
            while count < n_samples:
                if font_weights:
                    font_path = random.choices(font_list, weights=font_weights, k=1)[0]
                else:
                    font_path = random.choice(font_list)
                
                text = random_text(corpus_prob=corpus_prob, word_prob=word_split_prob, phrase_prob=phrase_split_prob)
                if not text:
                    continue

                img = generate_sample(
                    text, font_path,
                    font_size=random.randint(*font_size_range),
                )
                
                if img is None:
                    continue

                img_path = out / f"{worker_id:02d}_{count:07d}.png"
                cv2.imwrite(str(img_path), img)
                
                # Ghi trực tiếp nhãn vào file
                m = {"file": f"{worker_id:02d}_{count:07d}.png", "label": text}
                f_json.write(json.dumps(m, ensure_ascii=False) + "\n")
                f_json.flush()  # Ép xuống đĩa để tránh mất dữ liệu
                
                count += 1
                pbar.update(1)
