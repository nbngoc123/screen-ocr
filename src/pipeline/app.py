"""
pipeline.py — Full end-to-end OCR pipeline.

Kết hợp mss (chụp ảnh) -> DBNet++ (detect) -> CRNN (recognize) -> postprocess (chuẩn hóa).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.detector import BoundingBox, DBNetDetector
from src.recognizer.postprocess import postprocess
from src.recognizer.core import CRNNRecognizer

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    box: BoundingBox
    text: str
    conf: float


class OCRPipeline:
    """
    End-to-end OCR Pipeline.
    """
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Khởi tạo pipeline với detector và recognizer.
        """
        self.config = config
        
        # Load configs
        det_path = config["paths"]["det_model"]
        rec_path = config["paths"]["rec_model"]
        charset_path = config["paths"]["charset"]
        providers = config["inference"]["providers"]
        
        self.conf_threshold = config["inference"]["conf_threshold"]
        self.min_box_area = config["inference"]["min_box_area"]
        
        # Khởi tạo models (Singleton)
        self.detector = DBNetDetector.get_instance(det_path, providers)
        self.recognizer = CRNNRecognizer.get_instance(rec_path, charset_path, providers)

    def run(self, image: np.ndarray) -> list[OCRResult]:
        """
        Chạy toàn bộ pipeline trên một ảnh numpy (BGR).
        
        Args:
            image: Ảnh đầu vào.
            
        Returns:
            Danh sách kết quả OCR (bbox, text, conf).
        """
        # TODO: implement
        raise NotImplementedError
