import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.api.inference import OCRInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Screen OCR API", description="API nhận dạng chữ viết từ ảnh (CRNN)")

# Khởi tạo model một lần duy nhất khi server chạy
ocr_engine = None

@app.on_event("startup")
async def startup_event():
    global ocr_engine
    logger.info("Đang nạp mô hình OCR vào bộ nhớ (Sử dụng GPU nếu có)...")
    try:
        ocr_engine = OCRInference()
        logger.info("Nạp mô hình thành công!")
    except Exception as e:
        logger.error(f"Lỗi khi nạp mô hình: {e}")
        raise RuntimeError(f"Không thể khởi tạo OCR Inference: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên phải là hình ảnh.")
        
    start_time = time.time()
    
    try:
        image_bytes = await file.read()
        results = ocr_engine.predict_image(image_bytes)
        
        elapsed = time.time() - start_time
        return JSONResponse(content={
            "status": "success",
            "results": results,
            "inference_time_ms": round(elapsed * 1000, 2)
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán ảnh: {e}")
        raise HTTPException(status_code=500, detail=str(e))
