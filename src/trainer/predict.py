"""
predict.py — Script dự đoán chữ từ ảnh bằng model CRNN đã train.
"""
import argparse
import sys
import os
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import cv2
import matplotlib.pyplot as plt

from src.dataset.charset import CharsetCodec
from src.data_generation.augment import preprocess_for_crnn
from src.recognizer.model import CRNN, ctc_greedy_decode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn tới ảnh cần nhận dạng")
    parser.add_argument("--weights", type=str, default="checkpoints/crnn_best.pth", help="File trọng số model")
    parser.add_argument("--charset", type=str, default="data/charset.txt", help="File charset")
    parser.add_argument("--plot", action="store_true", help="Hiển thị ảnh và kết quả lên màn hình")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Charset
    codec = CharsetCodec(args.charset)
    
    # 2. Khởi tạo Model & Load Weights
    model = CRNN(num_classes=len(codec)).to(device)
    if not os.path.exists(args.weights):
        print(f"[!] Không tìm thấy file weights tại {args.weights}")
        return
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"[+] Đã load model thành công từ {args.weights}")

    # 3. Đọc và Tiền xử lý ảnh
    if not os.path.exists(args.image):
        print(f"[!] Không tìm thấy ảnh tại {args.image}")
        return
        
    img = cv2.imread(args.image)
    if img is None:
        print("[!] Không thể đọc ảnh, file có thể bị hỏng.")
        return
        
    # Tiền xử lý ảnh (Resize Height=32, Normalization...)
    img_tensor_np = preprocess_for_crnn(img, target_h=32, is_train=False)
    
    # Chuyển numpy (H, W) -> tensor (1, 1, H, W)
    img_tensor = torch.from_numpy(img_tensor_np).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # 4. Dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        preds = ctc_greedy_decode(outputs, codec.charset)
        result_text = preds[0]
        
    print("-" * 50)
    print(f"File ảnh: {args.image}")
    print(f"Kết quả nhận dạng (Prediction): >>> {result_text} <<<")
    print("-" * 50)
    
    # 5. Trực quan hoá (tuỳ chọn)
    if args.plot:
        plt.figure(figsize=(10, 3))
        # Chuyển BGR sang RGB để vẽ matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"Dự đoán: {result_text}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
