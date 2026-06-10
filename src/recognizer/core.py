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

from src.dataset import CharsetCodec

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
        target_h: int = 64,
        providers: list[str] | None = None
    ) -> None:
        """
        Khởi tạo ONNX session. Nên dùng get_instance().
        """
        if not HAS_ORT:
            raise ImportError("onnxruntime chưa được cài đặt")
            
        self.codec = CharsetCodec(charset_path)
        self.target_h = target_h
        self._lock = Lock()
        
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        logger.info(f"Loading CRNN from {model_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    @classmethod
    def get_instance(
        cls, 
        model_path: str, 
        charset_path: str,
        target_h: int = 64,
        providers: list[str] | None = None
    ) -> CRNNRecognizer:
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(model_path, charset_path, target_h, providers)
            return cls._instance

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize về target_h, grayscale, normalize."""
        h, w = image.shape[:2]
        
        # Resize chiều cao về target_h, giữ tỉ lệ
        new_w = int(w * (self.target_h / float(h)))
        new_w = max(4, new_w)  # Không cho quá nhỏ
        
        img = cv2.resize(image, (new_w, self.target_h), interpolation=cv2.INTER_AREA)
        
        # Chuyển sang ảnh xám nếu cần
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Chuẩn hoá [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # Thêm các chiều (Batch=1, Channel=1) -> (1, 1, target_h, W)
        return img[np.newaxis, np.newaxis, :, :]

    def _decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Giải mã logits dạng (1, W, num_classes) thành chuỗi."""
        # CỰC KỲ QUAN TRỌNG: logits là raw score, cần đi qua Softmax để thành xác suất [0, 1]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs_array = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Lấy class có xác suất cao nhất tại mỗi time step
        preds = np.argmax(probs_array, axis=2)[0]  # Shape: (W,)
        probs = np.max(probs_array, axis=2)[0]
        
        char_list = []
        conf_list = []
        for i, char_idx in enumerate(preds):
            if char_idx != 0 and (not (i > 0 and char_idx == preds[i - 1])):
                char_list.append(self.codec.charset[char_idx - 1])
                conf_list.append(probs[i])
                
        text = "".join(char_list)
        conf = float(np.mean(conf_list)) if conf_list else 0.0
        return text, conf

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Nhận diện chữ trên một ảnh crop.
        """
        inp = self._preprocess(crop)
        
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        
        logits = self.sess.run([output_name], {input_name: inp})[0]
        return self._decode(logits)

    def recognize_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """
        Nhận diện batch nhiều crop cùng lúc (pad về cùng width).
        """
        # CRNN chạy tuần tự hoặc padding batch
        # Ở đây đơn giản hoá xử lý bằng cách chạy từng ảnh, 
        # do batching chiều width khác nhau cần padding lằng nhằng
        results = []
        for crop in crops:
            results.append(self.recognize(crop))
        return results
