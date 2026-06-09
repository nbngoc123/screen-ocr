"""
recognizer.py — Text Recognition với CRNN ONNX.

Nhận input là crop ảnh từ detector, xuất ra chuỗi text.
Singleton pattern, batch inference hỗ trợ.
"""
from __future__ import annotations

import logging
from threading import Lock
from typing import Protocol

import cv2
import numpy as np

from src.charset import CharsetCodec

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

logger = logging.getLogger(__name__)


class TextRecognizer(Protocol):
    """Protocol bắt buộc cho mọi text recognizer."""
    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        ...
    
    def recognize_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        ...


class CRNNRecognizer:
    """
    CRNN ONNX inference.
    Singleton pattern theo screen-ocr-rules.md §1.2.
    Batch inference theo §6.1.
    """
    _instance: CRNNRecognizer | None = None
    _init_lock = Lock()

    def __init__(
        self, 
        model_path: str, 
        charset_path: str, 
        providers: list[str] | None = None
    ) -> None:
        """
        Khởi tạo ONNX session. Nên dùng get_instance().
        """
        if not HAS_ORT:
            raise ImportError("onnxruntime chưa được cài đặt")
            
        self.codec = CharsetCodec(charset_path)
        self._lock = Lock()
        
        # Placeholder
        self.sess = None
        # logger.info(f"Loading CRNN from {model_path}")
        # self.sess = ort.InferenceSession(model_path, providers=providers)

    @classmethod
    def get_instance(
        cls, 
        model_path: str, 
        charset_path: str,
        providers: list[str] | None = None
    ) -> CRNNRecognizer:
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(model_path, charset_path, providers)
            return cls._instance

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize về H=32, grayscale, normalize."""
        # TODO: implement
        raise NotImplementedError

    def _decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy decode và tính confidence score."""
        # TODO: implement
        raise NotImplementedError

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Nhận diện chữ trên một ảnh crop.
        """
        # TODO: implement
        raise NotImplementedError

    def recognize_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """
        Nhận diện batch nhiều crop cùng lúc (pad về cùng width).
        """
        # TODO: implement
        raise NotImplementedError
