import os
import torch
import cv2
import numpy as np
import yaml
import onnxruntime as ort

from src.detector.core import DBNetDetector
from src.recognizer.model import CRNN, ctc_greedy_decode
from src.trainer.train_cer import CharsetCodec
from src.data_generation.augment import preprocess_for_crnn

class OCRInference:
    def __init__(self, config_path: str = "configs/default.yaml", onnx_path: str = "checkpoints/crnn.onnx"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cấu hình Detector (DBNet)
        det_path = self.config['paths']['det_model']
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình DBNet tại: {det_path}")
        # Ép DBNet chạy CPU để tránh giành giật VRAM với PyTorch CRNN (giờ đều dùng ONNX)
        self.detector = DBNetDetector.get_instance(det_path, providers=["CPUExecutionProvider"])
        
        # Cấu hình Recognizer (CRNN ONNX)
        self.target_h = self.config['preprocess']['target_h']
        charset_path = self.config['paths']['charset']
        self.codec = CharsetCodec(charset_path)
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Không tìm thấy file ONNX: {onnx_path}")
            
        providers = self.config.get('inference', {}).get('providers', ['CPUExecutionProvider'])
        self.model = ort.InferenceSession(onnx_path, providers=providers)
        
    def predict_image(self, image_bytes: bytes) -> list[dict]:
        """
        Dự đoán OCR trên ảnh lớn.
        Trả về danh sách các vùng văn bản phát hiện được.
        """
        # Giải mã ảnh
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Không thể đọc được ảnh (invalid format)")
            
        # Bước 1: Dò tìm khung chữ bằng DBNet
        boxes = self.detector.detect(img)
        
        # Sắp xếp các box từ trên xuống dưới, từ trái qua phải
        boxes = sorted(boxes, key=lambda b: (b.y1, b.x1))
        
        results = []
        for box in boxes:
            # Lấy toạ độ
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            
            # Cắt ảnh (đảm bảo không bị âm)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue
                
            # Bước 2: Nhận dạng chữ trên từng vùng cắt
            img_tensor = preprocess_for_crnn(crop_img, self.target_h, is_train=False)
            
            # ONNX expects numpy array, shape (B, C, H, W) -> (1, 1, target_h, W)
            ort_input = np.expand_dims(img_tensor, axis=(0, 1)).astype(np.float32)
            
            # Chạy model ONNX
            input_name = self.model.get_inputs()[0].name
            out = self.model.run(None, {input_name: ort_input})[0]
            
            # Decode (chuyển sang tensor để dùng lại hàm ctc_greedy_decode cũ)
            out_tensor = torch.from_numpy(out)
            preds = ctc_greedy_decode(out_tensor, self.codec.charset)
                
            text = preds[0] if preds else ""
            if text.strip():
                results.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "text": text,
                    "confidence": float(box.confidence)
                })
                
        return results
