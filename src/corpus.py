"""
corpus.py — Text corpus cho synthetic data generation.

Cung cấp các chuỗi text theo domain (UI labels, câu tiếng Việt/Anh,
code snippets, số) để render lên ảnh synthetic.
"""
from __future__ import annotations

import random
import string

# ─── Domain corpus ────────────────────────────────────────────────────────────
# Xem week1-data-pipeline.md §2.1 để biết danh sách đầy đủ.
DOMAINS: dict[str, list[str]] = {
    "ui_labels": [
        # TODO: điền danh sách UI labels (OK, Cancel, Settings, Cài đặt, ...)
    ],
    "sentences_vn": [
        # TODO: điền câu tiếng Việt thường gặp trên screen
    ],
    "sentences_en": [
        # TODO: điền câu tiếng Anh thường gặp trên screen
    ],
    "code_snippets": [
        # TODO: điền code snippets ngắn
    ],
    "numbers": [
        # TODO: điền các dạng số (tiền tệ, ngày, hex, ...)
    ],
}


def random_text(min_len: int = 2, max_len: int = 40) -> str:
    """
    Sinh text ngẫu nhiên từ corpus hoặc generate tổng hợp.

    Xác suất 70% lấy từ corpus, 30% generate ngẫu nhiên.

    Args:
        min_len: Độ dài tối thiểu.
        max_len: Độ dài tối đa (text bị truncate nếu dài hơn).

    Returns:
        Chuỗi text, đã strip và truncate.
    """
    # TODO: implement — xem week1-data-pipeline.md §2.1
    raise NotImplementedError
