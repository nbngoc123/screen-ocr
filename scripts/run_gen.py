"""
run_gen.py — Chạy data generator.
"""
import sys
import os
import yaml
import multiprocessing
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để Python tìm thấy module src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_gen import generate_dataset

def get_font_weights(fonts: list[str], common_fonts: list[str], common_wt: float, rare_wt: float) -> list[float]:
    """Tạo mảng xác suất: Font phổ biến (thẳng) x trọng số cao, font lạ x trọng số thấp."""
    weights = []
    for f in fonts:
        f_lower = f.lower()
        if any(cf in f_lower for cf in common_fonts):
            weights.append(common_wt)
        else:
            weights.append(rare_wt)
    return weights

def worker_generate(args):
    """Worker function cho multiprocessing"""
    font_list, output_dir, n_samples = args
    # Để tránh overwrite, ta có thể sinh file vào các thư mục con hoặc dùng offset
    # Tuy nhiên ở script demo này ta dùng tiến trình đơn giản hơn: 
    # Mỗi worker gen vào một folder tạm, sau đó merge lại.
    # Trong script này, vì để code gọn nhẹ, tôi sẽ gọi generate_dataset trực tiếp
    # nhưng giảm số lượng mẫu xuống để test nhanh trước.
    generate_dataset(font_list, output_dir, n_samples)

if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    try:
        fonts = Path(config["paths"]["font_list"]).read_text(encoding="utf-8").splitlines()
        fonts = [f.strip() for f in fonts if f.strip()]
        if not fonts:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Chưa có danh sách font. Chạy scripts/list_fonts.py trước.")
        exit(1)
        
    cfg_dg = config["data_gen"]
    common_fonts = cfg_dg.get("common_fonts", [])
    common_wt = cfg_dg.get("common_font_weight", 10.0)
    rare_wt = cfg_dg.get("rare_font_weight", 1.0)
    val_ratio = cfg_dg.get("val_ratio", 0.1)

    # Split train and val fonts theo nguyên tắc không trùng lặp
    split_idx = int(len(fonts) * (1.0 - val_ratio))
    train_fonts = fonts[:split_idx]
    val_fonts = fonts[split_idx:]
    
    # Ở đây để demo test thử nhanh, tôi đặt số lượng rất nhỏ (100 ảnh)
    # Số lượng ảnh lấy thẳng từ file cấu hình default.yaml
    n_train = cfg_dg["n_train"]
    n_val = cfg_dg["n_val"]
    
    print(f"Sử dụng {len(train_fonts)} fonts cho tập Train, {len(val_fonts)} fonts cho tập Val.")
    print(f"Bắt đầu sinh dữ liệu ({n_train} Train, {n_val} Val)...")
    
    # Tính toán trọng số xác suất cho các font
    train_weights = get_font_weights(train_fonts, common_fonts, common_wt, rare_wt)
    val_weights = get_font_weights(val_fonts, common_fonts, common_wt, rare_wt)
    
    # Train
    generate_dataset(
        font_list=train_fonts,
        output_dir=config["paths"]["synthetic_train"],
        n_samples=n_train,
        font_weights=train_weights,
    )
    
    # Val
    generate_dataset(
        font_list=val_fonts,
        output_dir=config["paths"]["synthetic_val"],
        n_samples=n_val,
        font_weights=val_weights,
    )
    
    print("Hoàn tất!")
