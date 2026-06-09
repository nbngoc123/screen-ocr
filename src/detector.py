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
import yaml
from pathlib import Path

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

    def __init__(self, model_path: str, providers: list[str] | None = None, config_path: str = "configs/default.yaml") -> None:
        """
        Khởi tạo ONNX session. Không nên gọi trực tiếp, dùng get_instance().
        """
        if not HAS_ORT:
            raise ImportError("onnxruntime chưa được cài đặt")
        
        self.model_path = model_path
        self._lock = Lock()
        
        # Tự động đọc cấu hình detector từ file YAML
        self.config = {}
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get("detector", {})
        
        if providers is None:
            providers = ["CPUExecutionProvider"]
            
        logger.info(f"Loading DBNet from {model_path} with {providers}")
        self.sess = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

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

    def _preprocess(self, image: np.ndarray, max_size: int = 960) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Tiền xử lý ảnh:
        1. Resize để có chiều rộng/cao là bội số của 32. Giữ nguyên tỷ lệ hoặc giới hạn kích thước tối đa.
        2. Normalize theo ImageNet (mean, std).
        3. Đổi dạng NCHW.
        """
        h, w = image.shape[:2]
        
        # Để nhanh và dễ tính, giới hạn kích thước ảnh tối đa (max_size)
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            
        # Tính toán chiều cao, rộng mới là bội số của 32
        new_h = int(np.round(h * scale / 32) * 32)
        new_w = int(np.round(w * scale / 32) * 32)
        
        # Bảo vệ nếu ảnh quá nhỏ
        new_h = max(32, new_h)
        new_w = max(32, new_w)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Normalize
        c_mean = self.config.get("normalize_mean", [0.485, 0.456, 0.406])
        c_std = self.config.get("normalize_std", [0.229, 0.224, 0.225])
        mean = np.array(c_mean, dtype=np.float32)
        std = np.array(c_std, dtype=np.float32)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        
        # NCHW
        img_tensor = img_norm.transpose(2, 0, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0) # (1, 3, H, W)
        
        return img_tensor, (h, w)

    def _postprocess(self, prob_map: np.ndarray, orig_shape: tuple[int, int], thresh: float = 0.3, unclip_ratio: float = 2.0, min_area: int = 10) -> list[BoundingBox]:
        """
        Hậu xử lý Probability Map -> BoundingBoxes.
        - unclip_ratio: Hệ số nới lỏng (padding) box. Càng lớn thì box càng rộng. 
                        Chuẩn của DBNet thường là 1.5 - 2.0.
        """
        orig_h, orig_w = orig_shape
        pred_h, pred_w = prob_map.shape
        
        # Binarize
        bitmap = (prob_map > thresh).astype(np.uint8) * 255
        
        # Tìm contours
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Bỏ qua các contour quá nhỏ
            if area < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Tính tỉ lệ phục hồi lại kích thước gốc
            scale_x = orig_w / pred_w
            scale_y = orig_h / pred_h
            
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)
            
            # --- PADDING (UNCLIP) ---
            # Áp dụng công thức giãn nở chuẩn của DBNet: D = area * unclip_ratio / perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                dist = area * unclip_ratio / perimeter
                pad_x = int(dist * scale_x)
                pad_y = int(dist * scale_y)
            else:
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.1)
                
            # Thêm cấu hình pad cố định một chút xíu nữa để đảm bảo không bị lẹm viền chữ
            pad_x += 2
            pad_y += 2
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(orig_w, x2 + pad_x)
            y2 = min(orig_h, y2 + pad_y)
            
            # Tính confidence trung bình trong vùng chữ nhật (trên prob_map)
            score = np.mean(prob_map[y:y+h, x:x+w])
            
            boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(score)))
            
        return boxes

    def detect(self, image: np.ndarray, max_size: int | None = None, prob_threshold: float | None = None, unclip_ratio: float | None = None, min_area: int | None = None) -> list[BoundingBox]:
        """
        Phát hiện text bounding boxes. Nếu không truyền các tham số, hệ thống sẽ tự động dùng giá trị đọc từ file config YAML.
        """
        if image is None or image.size == 0:
            return []
            
        # Lấy tham số (ưu tiên tham số truyền vào hàm -> lấy từ file config -> lấy giá trị cứng mặc định)
        c_max_size = max_size if max_size is not None else self.config.get("max_size", 960)
        c_prob_thresh = prob_threshold if prob_threshold is not None else self.config.get("prob_threshold", 0.3)
        c_unclip = unclip_ratio if unclip_ratio is not None else self.config.get("unclip_ratio", 2.0)
        c_min_area = min_area if min_area is not None else self.config.get("min_area", 10)
            
        # 1. Preprocess
        input_tensor, orig_shape = self._preprocess(image, max_size=c_max_size)
        
        # 2. Inference
        with self._lock:
            outputs = self.sess.run([self.output_name], {self.input_name: input_tensor})
            
        # Probability map có shape (1, 1, H, W)
        prob_map = outputs[0][0, 0, :, :]
        
        # 3. Postprocess
        boxes = self._postprocess(prob_map, orig_shape, thresh=c_prob_thresh, unclip_ratio=c_unclip, min_area=c_min_area)
        return boxes
