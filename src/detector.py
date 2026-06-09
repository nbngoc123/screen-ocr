"""
detector.py — Text Detection với DBNet++.

Sử dụng ONNX Runtime để load mô hình DBNet++ phát hiện bounding box của text.
Theo rule 1.1, module này độc lập hoàn toàn với recognizer.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


class TextDetector(Protocol):
    """Protocol bắt buộc cho mọi text detector."""
    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        ...


class DBNetDetector:
    """
    DBNet++ ONNX inference.
    Singleton pattern theo screen-ocr-rules.md §1.2.
    Thread-safe inference theo §4.4.
    """
    _instance: DBNetDetector | None = None
    _init_lock = Lock()

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """
        Khởi tạo ONNX session. Không nên gọi trực tiếp, dùng get_instance().
        
        Args:
            model_path: Đường dẫn tới .onnx file (Lưu ý: hiện chưa có file này).
            providers: Execution providers (e.g., CUDAExecutionProvider).
        """
        if not HAS_ORT:
            raise ImportError("onnxruntime chưa được cài đặt")
        
        self.model_path = model_path
        self._lock = Lock()
        
        # Placeholder cho onnxruntime session
        self.sess = None
        # logger.info(f"Loading DBNet from {model_path}")
        # self.sess = ort.InferenceSession(model_path, providers=providers)

    @classmethod
    def get_instance(
        cls, 
        model_path: str, 
        providers: list[str] | None = None
    ) -> DBNetDetector:
        """Lấy singleton instance của DBNetDetector."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(model_path, providers)
            return cls._instance

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """
        Phát hiện text bounding boxes.
        
        Args:
            image: Ảnh đầu vào BGR.
            
        Returns:
            Danh sách BoundingBox.
        """
        # TODO: implement
        # 1. Preprocess (resize, normalize)
        # 2. lock -> sess.run
        # 3. Postprocess (từ probability map -> bounding boxes)
        raise NotImplementedError
