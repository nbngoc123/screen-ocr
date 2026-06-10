"""
debug_pipeline.py — Script test và debug trực tiếp OCR Pipeline.
"""
import sys
import os
import yaml
import cv2
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.app import OCRPipeline

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def debug_image(image_path: str):
    print(f"--- ĐANG DEBUG ẢNH: {image_path} ---")
    if not os.path.exists(image_path):
        print("Lỗi: Không tìm thấy ảnh!")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print("Lỗi: Không thể đọc ảnh bằng OpenCV.")
        return
        
    print(f"Kích thước ảnh: {img.shape}")
    
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Ép dùng CPU để không bị CUDA Out of memory khi debug
    config["inference"]["providers"] = ["CPUExecutionProvider"]
        
    print("Đang khởi tạo Pipeline...")
    pipeline = OCRPipeline(config)
    
    print("\n--- BƯỚC 1: DBNet Detection ---")
    # Debug trực tiếp detector
    bboxes = pipeline.detector.detect(img)
    print(f"DBNet tìm thấy {len(bboxes)} bounding boxes ban đầu.")
    for i, b in enumerate(bboxes[:5]):
        print(f"  Box {i+1}: ({b.x1}, {b.y1}) -> ({b.x2}, {b.y2})")
    if len(bboxes) > 5:
        print(f"  ... và {len(bboxes)-5} box khác.")
        
    if not bboxes:
        print("-> DBNet KHÔNG tìm thấy bất kỳ box nào. Có thể do prob_threshold hoặc box_score_threshold quá cao!")
        return
        
    print("\n--- BƯỚC 2: Lọc Box theo diện tích ---")
    valid_boxes = []
    for b in bboxes:
        area = (b.x2 - b.x1) * (b.y2 - b.y1)
        if area >= pipeline.min_box_area:
            valid_boxes.append(b)
    print(f"Sau khi lọc (min_box_area={pipeline.min_box_area}), còn lại {len(valid_boxes)} boxes.")
    
    if not valid_boxes:
        print("-> Tất cả các box đều quá nhỏ và bị lọc mất!")
        return
        
    print("\n--- BƯỚC 3: CRNN Recognition ---")
    results = pipeline.run(img)
    print(f"Pipeline trả về tổng cộng {len(results)} kết quả OCR (đã lọc conf >= {pipeline.conf_threshold}).")
    
    for i, res in enumerate(results):
        print(f"Kết quả {i+1}: Text='{res.text}' | Conf={res.conf:.4f} | Box=({res.box.x1},{res.box.y1})->({res.box.x2},{res.box.y2})")

    print("\n--- BƯỚC 4: Kết quả BỊ LỌC do Confidence < 0.6 ---")
    rejected = 0
    for b in valid_boxes:
        x1 = max(0, b.x1)
        y1 = max(0, b.y1)
        x2 = min(img.shape[1], b.x2)
        y2 = min(img.shape[0], b.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if i == 0:
            cv2.imwrite("debug_crop_1.png", crop)
            print("Đã lưu crop đầu tiên thành debug_crop_1.png")
            
        text, conf = pipeline.recognizer.recognize(crop)
        if conf < pipeline.conf_threshold or not text:
            print(f"  Bị lọc: Text='{text}' | Conf={conf:.4f} | Box=({b.x1},{b.y1})")
            rejected += 1
    print(f"Tổng số bị lọc: {rejected}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python scripts/debug_pipeline.py <đường_dẫn_ảnh>")
    else:
        debug_image(sys.argv[1])
