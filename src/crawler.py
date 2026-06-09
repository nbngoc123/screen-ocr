"""
crawler.py — UIAutomation screen crawler.

Thu thập real screen data tự động bằng Windows UIAutomation API.
Ground truth ~99% chính xác (lấy từ ctrl.Name của UI element).

Chạy trong background thread trong khi dùng máy bình thường.

Quy tắc (screen-ocr-rules.md §2.1):
    UIAutomation API cho độ tin cậy ~99%.
    Verify bằng visual inspection 1% random sample.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Event, Thread

import cv2
import mss
import numpy as np

try:
    import uiautomation as auto
    HAS_UIA = True
except ImportError:
    HAS_UIA = False
    logging.warning("uiautomation chưa cài — chạy: pip install uiautomation")

logger = logging.getLogger(__name__)


class ScreenCrawler:
    """
    Thu thập cặp (ảnh crop, text label) từ UI elements trên màn hình.

    Output format:
        {output_dir}/{idx:07d}.png   — ảnh crop của UI element
        {output_dir}/{idx:07d}.json  — {"label": str, "box": [...], "control_type": str}

    Example:
        crawler = ScreenCrawler(output_dir="data/real", max_samples=50_000)
        thread = crawler.run_background(interval_sec=20)
        # ... dùng máy bình thường ...
        crawler.stop()
    """

    def __init__(
        self,
        output_dir: str = "data/real",
        max_samples: int = 50_000,
        min_text_len: int = 2,
        min_box_area: int = 200,
    ) -> None:
        # TODO: implement — xem week1-data-pipeline.md §4.1
        raise NotImplementedError

    def collect_once(self) -> int:
        """
        Chụp màn hình hiện tại và thu thập toàn bộ UI elements.

        Returns:
            Số samples đã thu thập trong lần này.
        """
        # TODO: implement
        raise NotImplementedError

    def run_background(self, interval_sec: float = 30.0) -> Thread:
        """
        Chạy crawl liên tục trong background daemon thread.

        Args:
            interval_sec: Khoảng cách giữa các lần crawl (giây).

        Returns:
            Thread object đang chạy (daemon=True).
        """
        # TODO: implement
        raise NotImplementedError

    def stop(self) -> None:
        """Gửi signal dừng cho background thread."""
        # TODO: implement
        raise NotImplementedError
