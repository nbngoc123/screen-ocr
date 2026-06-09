"""
postprocess.py — Xử lý text sau nhận diện.

Chuẩn hóa khoảng trắng, sửa lỗi OCR phổ biến, format đặc thù.
"""
from __future__ import annotations

import re


def postprocess(text: str) -> str:
    """
    Chuẩn hóa output OCR.
    
    Các bước:
    1. Whitespace normalize
    2. Fix các lỗi phổ biến (0 <-> O)
    3. Custom rules
    """
    # 1. Whitespace normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. OCR common errors
    # Ví dụ: sửa số 0 nằm giữa chữ cái in hoa thành chữ O
    text = re.sub(r'\b([A-Z]{2,})0([A-Z0-9])', r'\g<1>O\g<2>', text)
    # Số O kẹp giữa các số thành số 0
    text = re.sub(r'(?<=[0-9])O(?=[0-9])', '0', text)
    
    # 3. Domain-specific fixes (email, urls...)
    text = re.sub(
        r'([\w.+\-]+)\s*@\s*([\w.\-]+\s*\.\s*[a-z]{2,})',
        lambda m: m.group(1) + '@' + m.group(2).replace(' ', ''),
        text
    )
    
    return text
