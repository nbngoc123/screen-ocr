from __future__ import annotations
import logging
from threading import Lock
import numpy as np
import yaml
from pathlib import Path

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from .types import BoundingBox
from .preprocess import preprocess_image
from .postprocess import postprocess_dbnet

logger = logging.getLogger(__name__)

class DBNetDetector:
    """
    DBNet++ ONNX inference.
    Singleton pattern theo screen-ocr-rules.md §1.2.
    Thread-safe inference theo §4.4.
    """
    _instance: 'DBNetDetector' | None = None
    _init_lock = Lock()

    def __init__(self, model_path: str, providers: list[str] | None = None, config_path: str = "configs/default.yaml") -> None:
        if not HAS_ORT:
            raise ImportError("onnxruntime chưa được cài đặt")
        
        self.model_path = model_path
        self._lock = Lock()
        
        self.config = {}
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get("detector", {})
        
        if providers is None:
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            providers = full_config.get("inference", {}).get("providers", ["CPUExecutionProvider"])
            
        logger.info(f"Loading DBNet from {model_path} with {providers}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    @classmethod
    def get_instance(cls, model_path: str, providers: list[str] | None = None) -> 'DBNetDetector':
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(model_path, providers)
            return cls._instance

    def detect(self, image: np.ndarray, max_size: int | None = None, prob_threshold: float | None = None, unclip_ratio: float | None = None, min_area: int | None = None) -> list[BoundingBox]:
        if image is None or image.size == 0:
            return []
            
        c_max_size = max_size if max_size is not None else self.config.get("max_size", 960)
        c_prob_thresh = prob_threshold if prob_threshold is not None else self.config.get("prob_threshold", 0.3)
        c_unclip = unclip_ratio if unclip_ratio is not None else self.config.get("unclip_ratio", 2.0)
        c_min_area = min_area if min_area is not None else self.config.get("min_area", 10)
        c_min_pad_x = self.config.get("min_padding_x", 4)
        c_min_pad_y = self.config.get("min_padding_y", 4)
        c_box_score = self.config.get("box_score_threshold", 0.5)
            
        input_tensor, orig_shape = preprocess_image(image, self.config, max_size=c_max_size)
        
        with self._lock:
            outputs = self.sess.run([self.output_name], {self.input_name: input_tensor})
            
        out = outputs[0]
        if out.ndim == 4:
            prob_map = out[0, 0]
        elif out.ndim == 3:
            prob_map = out[0]
        else:
            raise RuntimeError(f"Unexpected DBNet output shape: {out.shape}")
        
        return postprocess_dbnet(
            prob_map, orig_shape, 
            thresh=c_prob_thresh, unclip_ratio=c_unclip, 
            min_area=c_min_area, min_padding_x=c_min_pad_x, 
            min_padding_y=c_min_pad_y, box_score_threshold=c_box_score
        )
