"""
predict.py — Script dự đoán chữ từ ảnh bằng model CRNN đã train.
Hỗ trợ dự đoán ảnh lẻ hoặc kiểm thử hàng loạt từ thư mục có kèm nhãn gốc.
"""
import argparse
import sys
import os
import json
import random
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

def predict_single(model, codec, img_path, device):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
        
    img_tensor_np = preprocess_for_crnn(img, target_h=32, is_train=False)
    img_tensor = torch.from_numpy(img_tensor_np).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        preds = ctc_greedy_decode(outputs, codec.charset)
        
    return preds[0], img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn tới ảnh lẻ hoặc thư mục chứa ảnh (vd: data/synthetic/val)")
    parser.add_argument("--weights", type=str, default="checkpoints/crnn_best.pth", help="File trọng số model")
    parser.add_argument("--charset", type=str, default="data/charset.txt", help="File charset")
    parser.add_argument("--num-samples", type=int, default=5, help="Số lượng ảnh test nếu input là thư mục")
    parser.add_argument("--plot", action="store_true", help="Hiển thị ảnh và kết quả lên màn hình")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharsetCodec(args.charset)
    model = CRNN(num_classes=len(codec)).to(device)
    
    if not os.path.exists(args.weights):
        print(f"[!] Không tìm thấy file weights tại {args.weights}")
        return
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"[+] Đã load model thành công từ {args.weights}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Đường dẫn không tồn tại: {args.input}")
        return

    samples = []
    
    # Nếu là thư mục, thử đọc labels.jsonl
    if input_path.is_dir():
        labels_file = input_path / "labels.jsonl"
        if labels_file.exists():
            with open(labels_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                random.shuffle(lines)
                for line in lines[:args.num_samples]:
                    m = json.loads(line)
                    img_path = input_path / m["file"]
                    samples.append((img_path, m["label"]))
        else:
            print("[!] Không tìm thấy file labels.jsonl trong thư mục.")
            return
    else:
        # Nếu là file lẻ
        samples.append((input_path, "Không có (chỉ dự đoán)"))

    # Chạy dự đoán
    print("-" * 60)
    results_for_plot = []
    for img_path, true_label in samples:
        pred_text, img = predict_single(model, codec, img_path, device)
        if pred_text is None:
            print(f"[!] Lỗi đọc ảnh {img_path.name}")
            continue
            
        print(f"File : {img_path.name}")
        print(f"Label: {true_label}")
        print(f"Pred : >>> {pred_text} <<<")
        print("-" * 60)
        
        results_for_plot.append((img, true_label, pred_text))

    # Trực quan hoá
    if args.plot and results_for_plot:
        n = len(results_for_plot)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n))
        if n == 1:
            axes = [axes]
            
        for ax, (img, true_label, pred_text) in zip(axes, results_for_plot):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"True: {true_label}  |  Pred: {pred_text}")
            ax.axis("off")
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
