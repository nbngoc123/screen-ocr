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
        target_h = config.get("preprocess", {}).get("target_h", 64)
        self.recognizer = CRNNRecognizer.get_instance(rec_path, charset_path, target_h, providers)

    def run(self, image: np.ndarray) -> list[OCRResult]:
        """
        Chạy toàn bộ pipeline trên một ảnh numpy (BGR).
        
        Args:
            image: Ảnh đầu vào.
            
        Returns:
            Danh sách kết quả OCR (bbox, text, conf).
        """
        if image is None or image.size == 0:
            return []

        # 1. Detect bboxes using DBNet
        bboxes = self.detector.detect(image)
        if not bboxes:
            return []

        # 2. Lọc box theo diện tích (nếu có min_box_area)
        valid_boxes = []
        for b in bboxes:
            area = (b.x2 - b.x1) * (b.y2 - b.y1)
            if area >= self.min_box_area:
                valid_boxes.append(b)
                
        # 3. Sort boxes: Hậu xử lý gom dòng (top-to-bottom) và sắp xếp (left-to-right)
        from src.pipeline.postprocess import sort_boxes_to_lines
        valid_boxes = sort_boxes_to_lines(valid_boxes)

        results = []
        for b in valid_boxes:
            # Lấy toạ độ an toàn (có thể pad thêm nếu muốn)
            x1 = max(0, b.x1)
            y1 = max(0, b.y1)
            x2 = min(image.shape[1], b.x2)
            y2 = min(image.shape[0], b.y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = image[y1:y2, x1:x2]
            
            # 4. Recognize text
            text, conf = self.recognizer.recognize(crop)
            
            # Áp dụng Hậu xử lý văn bản (sửa lỗi 0/O, khoảng trắng)
            text = postprocess(text)
            
            # 5. Lọc kết quả rỗng và threshold
            if text and conf >= self.conf_threshold:
                results.append(OCRResult(box=b, text=text, conf=conf))

        return results
