import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import DBNetDetector
import os

def test():
    print("Khởi tạo DBNetDetector...")
    detector = DBNetDetector.get_instance("data/models/dbnet.onnx")
    
    # Tạo một bức ảnh giả (chứa chữ) bằng OpenCV để test
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST DBNET DETECTOR", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "HELLO WORLD!", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    print("Đang chạy detect...")
    boxes = detector.detect(img)
    print(f"Phát hiện được {len(boxes)} bounding boxes:")
    for b in boxes:
        print(f" - Box: ({b.x1}, {b.y1}) -> ({b.x2}, {b.y2}), Score: {b.confidence:.2f}")
        cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
        
    out_path = "dbnet_test_output.png"
    cv2.imwrite(out_path, img)
    print(f"Đã lưu ảnh kết quả vào {out_path}")

if __name__ == "__main__":
    test()
