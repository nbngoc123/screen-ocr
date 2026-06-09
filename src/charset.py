"""
charset.py — Định nghĩa và quản lý character set cho CRNN.

Charset gồm ~235 ký tự: ASCII printable + tiếng Việt Unicode NFC + UI extras.
Index 0 được giữ làm CTC blank token.
"""
from __future__ import annotations

import string
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Ký tự tiếng Việt (Unicode NFC) ─────────────────────────────────────────
_VIET_LOWER = (
    "àáảãạ"
    "ăắằẳẵặ"
    "âấầẩẫậ"
    "èéẻẽẹ"
    "êếềểễệ"
    "ìíỉĩị"
    "òóỏõọ"
    "ôốồổỗộ"
    "ơớờởỡợ"
    "ùúủũụ"
    "ưứừửữự"
    "ỳýỷỹỵ"
    "đ"
)
_VIET_UPPER = _VIET_LOWER.upper()

# Ký tự UI phổ biến ngoài ASCII printable
_UI_EXTRAS = "…•·×÷±°©®™€£¥₫→←↑↓↔"


def build_charset() -> str:
    """
    Xây dựng charset đầy đủ cho bài toán Việt + Anh trên screen.

    Returns:
        Chuỗi ký tự đã sort, không trùng lặp (~235 chars).
    """
    # TODO: implement — xem week1-data-pipeline.md §1.3
    raise NotImplementedError


def save_charset(path: str = "data/charset.txt") -> None:
    """
    Sinh và lưu charset ra file.

    Args:
        path: Đường dẫn output, relative từ project root.
    """
    # TODO: implement
    raise NotImplementedError


def load_charset(path: str = "data/charset.txt") -> str:
    """
    Đọc charset từ file.

    Args:
        path: Đường dẫn charset file.

    Returns:
        Chuỗi ký tự charset.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    # TODO: implement
    raise NotImplementedError


class CharsetCodec:
    """
    Encode/decode string ↔ index list cho CTC loss.

    Index 0 = CTC blank (không dùng cho ký tự thực).
    Index 1..N = ký tự trong charset.

    Example:
        codec = CharsetCodec("data/charset.txt")
        encoded = codec.encode("Hello")   # [8, 5, 12, 12, 15]
        decoded = codec.decode(encoded)   # "Hello"
    """

    def __init__(self, charset_path: str = "data/charset.txt") -> None:
        # TODO: implement — load charset, build char2idx + idx2char dicts
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """
        Chuyển chuỗi text thành list index.
        Ký tự không có trong charset sẽ bị bỏ qua (OOV).

        Args:
            text: Chuỗi cần encode.

        Returns:
            List index tương ứng (không bao gồm blank).
        """
        # TODO: implement
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        """
        CTC collapse: loại blank (index 0) và ký tự lặp liên tiếp.

        Args:
            indices: List index từ model output.

        Returns:
            Chuỗi text đã decode.
        """
        # TODO: implement
        raise NotImplementedError

    def __len__(self) -> int:
        """Số ký tự trong charset (không tính blank)."""
        # TODO: implement
        raise NotImplementedError
