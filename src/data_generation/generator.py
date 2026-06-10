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
import glob
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .corpus import random_text

logger = logging.getLogger(__name__)

# Load BG_POOL
BG_POOL = []
for p in glob.glob("data/bg_pool/*.png"):
    img = cv2.imread(p)
    if img is not None:
        BG_POOL.append(img)

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


@lru_cache(maxsize=8192)
def _is_char_supported(char: str, font_path: str, size: int) -> bool:
    """Kiểm tra một ký tự có được support bởi font hay bị biến thành tofu."""
    if char.isspace():
        return True
    font = _load_font(font_path, size)
    # Dùng getmask — nếu mask toàn 0 (empty) thì font không support
    try:
        mask = font.getmask(char)
        return mask.getbbox() is not None  # None = empty mask = tofu
    except Exception:
        return False


def check_text_fits_font(text: str, font_path: str, size: int) -> bool:
    """Kiểm tra xem font có chứa đủ tất cả glyph cho text không."""
    for c in text:
        if not _is_char_supported(c, font_path, size):
            return False
    return True


def _generate_background(w: int, h: int, bg_light_prob: float = 0.60, bg_dark_prob: float = 0.20) -> np.ndarray:
    """Sinh ảnh nền đa dạng (nền sáng, nền tối, hoặc nền texture)."""
    mode = random.random()

    if mode < bg_light_prob:
        color = np.random.randint(180, 255, (h, w, 3), dtype=np.uint8)
    elif mode < bg_light_prob + bg_dark_prob:
        color = np.random.randint(20, 100, (h, w, 3), dtype=np.uint8)
    else:
        color = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        color = cv2.GaussianBlur(color, (31, 31), 0)

    return color


def generate_sample(
    text: str,
    font_path: str,
    font_size: int | None = None,
    padding: int | None = None,
    shadow_prob: float = 0.25,
    stroke_prob: float = 0.30,
    bg_light_prob: float = 0.60,
    bg_dark_prob: float = 0.20,
    real_bg_prob: float = 0.30,
) -> np.ndarray | None:
    """
    Sinh một ảnh chứa text với nhiều variation cho UI (bóng, viền, padding ngẫu nhiên).
    """
    size = font_size or random.randint(10, 52)
    padding = padding if padding is not None else random.randint(0, 8)
    
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

    # Sinh background ngẫu nhiên bằng _generate_background hoặc lấy từ BG_POOL
    if random.random() < real_bg_prob and BG_POOL:
        bg_img = random.choice(BG_POOL)
        bg_img = cv2.resize(bg_img, (w, h))
    else:
        bg_img = _generate_background(w, h, bg_light_prob, bg_dark_prob)
    
    # Ước lượng màu nền trung bình để đảm bảo contrast
    mean_bg = tuple(int(x) for x in bg_img.mean(axis=(0, 1)))

    # Sinh foreground ngẫu nhiên
    fg = _rand_color(0, 80) if random.random() < 0.8 else _rand_color(150, 255)
    fg = _ensure_contrast(mean_bg, fg)

    # Convert cv2 img sang PIL để vẽ
    img = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    x = padding
    y = padding - bbox[1]

    # Đổ bóng (Shadow)
    if random.random() < shadow_prob:
        shadow_color = (0, 0, 0)
        shadow_dx = random.randint(1, 2)
        shadow_dy = random.randint(1, 2)
        draw.text(
            (x + shadow_dx, y + shadow_dy),
            text,
            font=font,
            fill=shadow_color,
        )

    # Viền chữ (Stroke)
    stroke_width = 0
    stroke_fill = None

    if random.random() < stroke_prob:
        stroke_width = random.randint(1, 2)
        # Nền tối/viền trắng hoặc Nền sáng/viền đen
        if sum(fg) < 250:
            stroke_fill = (255, 255, 255)
        else:
            stroke_fill = (0, 0, 0)

    draw.text(
        (x, y),
        text,
        font=font,
        fill=fg,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    # Convert RGB to BGR cho OpenCV
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def generate_sample_multidpi(
    text: str, font_path: str, base_size: int,
    **kwargs
) -> np.ndarray | None:
    dpi_scale = random.choice([1.0, 1.25, 1.5, 2.0])
    render_size = int(base_size * dpi_scale)

    img = generate_sample(text, font_path, font_size=render_size, **kwargs)
    if img is None:
        return None

    if dpi_scale > 1.0:
        h, w = img.shape[:2]
        new_w = int(w / dpi_scale)
        new_h = int(h / dpi_scale)
        if new_w < 4 or new_h < 4:
            return None
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img

def generate_negative_sample() -> np.ndarray:
    """Ảnh không có text — icon crop, gradient, texture."""
    mode = random.choice(["solid", "gradient_sim", "noise", "ui_crop"])
    w = random.randint(40, 300)
    h = random.randint(12, 80)

    if mode == "noise":
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    elif mode == "gradient_sim":
        base = _rand_color(100, 220)
        img = np.full((h, w, 3), base, dtype=np.uint8)
        img = cv2.GaussianBlur(img + np.random.randint(-20, 20, img.shape, np.int16).clip(0,255).astype(np.uint8), (5,5), 0)
        return img
    elif mode == "ui_crop" and BG_POOL:
        bg_img = random.choice(BG_POOL)
        return cv2.resize(bg_img, (w, h))
    else:
        return np.full((h, w, 3), _rand_color(180, 255), dtype=np.uint8)

def _quality_check(img: np.ndarray, label: str) -> tuple[bool, str]:
    """Trả về (pass, reason) cho mỗi generated sample."""
    h, w = img.shape[:2]

    # Quá nhỏ
    if w < 20 or h < 8:
        return False, f"too_small_{w}x{h}"

    # Contrast quá thấp (foreground và background gần nhau)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = float(gray.std())
    if std < 8.0:
        return False, f"low_contrast_{std:.1f}"

    # Ảnh quá mờ (blur score)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 5.0:
        return False, f"too_blurry_{lap_var:.1f}"

    # Label quá ngắn sau khi strip (nếu text bị rỗng)
    # Bỏ qua cho sample negative (label "")
    if len(label) > 0 and len(label.strip()) < 1:
        return False, "empty_label"

    return True, "ok"

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
    hard_prob: float = 0.10,
    shadow_prob: float = 0.25,
    stroke_prob: float = 0.30,
    bg_light_prob: float = 0.60,
    bg_dark_prob: float = 0.20,
    real_bg_prob: float = 0.30,
    neg_prob: float = 0.05,
    multi_dpi_prob: float = 0.20,
) -> int:
    """Sinh toàn bộ synthetic dataset ra thư mục."""
    
    # Seed cho worker
    random.seed(worker_id + 42)
    np.random.seed(worker_id + 42)
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    rejected_by_font = 0
    quality_rejected = {}

    # Mở file ghi liên tục (append hoặc write) để không bị mất dữ liệu nếu user bấm Ctrl+C
    worker_label_file = out / f"labels_{worker_id}.jsonl"
    with open(worker_label_file, "w", encoding="utf-8") as f_json:
        with tqdm(total=n_samples, desc=f"Worker {worker_id}", unit="img", position=worker_id) as pbar:
            while count < n_samples:
                r = random.random()
                
                # 1. Negative Sample
                if r < neg_prob:
                    img = generate_negative_sample()
                    text = ""
                
                # 2. Normal Single Line
                else:
                    if font_weights:
                        font_path = random.choices(font_list, weights=font_weights, k=1)[0]
                    else:
                        font_path = random.choice(font_list)
                    
                    text = random_text(
                        corpus_prob=corpus_prob, 
                        word_prob=word_split_prob, 
                        phrase_prob=phrase_split_prob,
                        hard_prob=hard_prob
                    )
                    if not text:
                        continue

                    base_size = random.randint(*font_size_range)
                    if not check_text_fits_font(text, font_path, base_size):
                        rejected_by_font += 1
                        continue

                    if random.random() < multi_dpi_prob:
                        img = generate_sample_multidpi(
                            text, font_path, base_size,
                            shadow_prob=shadow_prob,
                            stroke_prob=stroke_prob,
                            bg_light_prob=bg_light_prob,
                            bg_dark_prob=bg_dark_prob,
                            real_bg_prob=real_bg_prob,
                        )
                    else:
                        img = generate_sample(
                            text, font_path,
                            font_size=base_size,
                            shadow_prob=shadow_prob,
                            stroke_prob=stroke_prob,
                            bg_light_prob=bg_light_prob,
                            bg_dark_prob=bg_dark_prob,
                            real_bg_prob=real_bg_prob,
                        )
                
                if img is None:
                    continue

                # Lọc chất lượng
                ok, reason = _quality_check(img, text)
                if not ok:
                    quality_rejected[reason] = quality_rejected.get(reason, 0) + 1
                    continue

                img_path = out / f"{worker_id:02d}_{count:07d}.png"
                cv2.imwrite(str(img_path), img)
                
                # Ghi trực tiếp nhãn vào file
                m = {"file": f"{worker_id:02d}_{count:07d}.png", "label": text}
                f_json.write(json.dumps(m, ensure_ascii=False) + "\n")
                f_json.flush()  # Ép xuống đĩa để tránh mất dữ liệu
                
                count += 1
                pbar.update(1)

    # In ra báo cáo thống kê cho worker này
    logger.info(f"Worker {worker_id} - Generated: {count} | Rejected by Font: {rejected_by_font} | Quality Rejected: {quality_rejected}")
    return count
