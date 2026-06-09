# scripts/visualize_aug.py
"""Kiểm tra augmentation visually (Week 1 - 3.2)."""
import sys
import os
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để Python tìm thấy module src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import generate_sample
from src.data_generation import preprocess_for_crnn

def main():
    font_file = Path("data/font_list.txt")
    if not font_file.exists():
        print("Chưa có danh sách font. Chạy scripts/list_fonts.py trước.")
        sys.exit(1)
        
    fonts = font_file.read_text(encoding="utf-8").splitlines()
    fonts = [f for f in fonts if f.strip()]

    fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    for row in axes:
        # Lặp đến khi tìm được font hỗ trợ (không bị lỗi tofu)
        orig = None
        while orig is None:
            font = random.choice(fonts)
            text = "Xin chào 123 OCR"
            orig = generate_sample(text, font, font_size=24)
            
        row[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        row[0].set_title("Original")
        for i in range(1, 6):
            # is_train=True kích hoạt toàn bộ hệ thống thêm noise, blur
            aug = preprocess_for_crnn(orig, is_train=True)
            
            # Vì aug đã normalize về [-1, 1], ta cần đưa về [0, 1] để plot
            aug_display = (aug * 0.5) + 0.5
            
            row[i].imshow(aug_display, cmap="gray")
            row[i].set_title(f"Aug {i}")
        for ax in row:
            ax.axis("off")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = reports_dir / "augmentation_check.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Đã lưu kết quả kiểm tra vào: {out_path}")

if __name__ == "__main__":
    main()
