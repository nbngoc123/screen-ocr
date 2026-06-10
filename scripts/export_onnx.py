"""
export_onnx.py — Export PyTorch model sang ONNX.
"""
import os
import sys
import yaml
from pathlib import Path
import torch

# Thêm thư mục gốc vào đường dẫn hệ thống để Python tìm thấy module src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recognizer.model import CRNN

# Ép Windows Terminal dùng UTF-8 để không bị lỗi UnicodeEncodeError
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def export_to_onnx():
    print("Bắt đầu quy trình xuất mô hình CRNN sang ONNX...")
    
    # Đọc config
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    charset_path = config["paths"]["charset"]
    best_weight_path = config["paths"]["best_weight"]
    out_onnx_path = config["paths"]["rec_model"]
    target_h = config.get("preprocess", {}).get("target_h", 64)
    
    if not os.path.exists(charset_path):
        print(f"Lỗi: Không tìm thấy file charset tại {charset_path}")
        return
        
    if not os.path.exists(best_weight_path):
        print(f"Lỗi: Không tìm thấy file weights tại {best_weight_path}")
        return

    from src.dataset.charset import load_charset
    charset = load_charset(charset_path)
    num_classes = len(charset)
    
    print(f"Số lượng ký tự (classes): {num_classes}")
    
    # Khởi tạo mô hình và nạp weights
    model = CRNN(num_classes=num_classes)
    print(f"Load weights từ: {best_weight_path}")
    state_dict = torch.load(best_weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Tạo dummy input (Batch=1, Kênh=1, Chiều cao=target_h, Chiều rộng=tùy ý)
    dummy_input = torch.randn(1, 1, target_h, 200)

    # Đảm bảo thư mục đầu ra tồn tại
    Path(out_onnx_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Xuất ONNX ra: {out_onnx_path}")
    
    # Export ONNX với chiều width (3) là dynamic
    torch.onnx.export(
        model, 
        dummy_input, 
        out_onnx_path, 
        export_params=True, 
        opset_version=14,  # Dùng opset 14 hỗ trợ tốt adaptive pool
        do_constant_folding=True, 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes={
            "input": {0: "batch_size", 3: "width"}, 
            "output": {0: "batch_size", 1: "seq_len"}
        }
    )
    
    print("Xuất ONNX thành công!")

if __name__ == "__main__":
    export_to_onnx()
