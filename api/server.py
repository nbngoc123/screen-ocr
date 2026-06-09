"""
server.py — FastAPI service cho Screen OCR.

Nhận base64 image qua HTTP POST, trả về text và boxes.
"""
from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import OCRPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="Screen OCR Engine API")

# Pipeline instance global
_pipeline: OCRPipeline | None = None

class OCRRequest(BaseModel):
    image_b64: str
    conf_threshold: float | None = None

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class OCRResponseItem(BaseModel):
    box: BBox
    text: str
    conf: float

@app.on_event("startup")
def load_models():
    """Load config và model khi khởi động server."""
    global _pipeline
    try:
        with open("configs/default.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        _pipeline = OCRPipeline(config)
        logger.info("OCR Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo pipeline: {e}")
        # Trong thực tế, DBNet++ model chưa có nên có thể lỗi khởi tạo
        # raise e

@app.post("/recognize", response_model=list[OCRResponseItem])
def recognize_endpoint(req: OCRRequest):
    """
    Nhận diện chữ từ ảnh base64.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng")
        
    try:
        # Decode base64
        img_bytes = base64.b64decode(req.image_b64)
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            raise ValueError("Ảnh base64 không hợp lệ")
            
        # TODO: Cập nhật config temporary (như conf_threshold) nếu có
        # Run pipeline
        # results = _pipeline.run(img_np)
        
        # Format output
        # return [{"box": {"x1": r.box.x1, "y1": r.box.y1, "x2": r.box.x2, "y2": r.box.y2}, "text": r.text, "conf": r.conf} for r in results]
        
        return []
    except Exception as e:
        logger.error(f"Lỗi inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
