"""
clean_data.py — Script dọn dẹp dữ liệu ảnh đã sinh.
"""
import yaml
import shutil
from pathlib import Path

def clean_dir(path: str):
    p = Path(path)
    if not p.exists():
        return
    count = 0
    for item in p.iterdir():
        # Bỏ qua file .gitkeep để không làm hỏng cấu trúc git
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
            count += 1
        else:
            item.unlink()
            count += 1
    print(f"Đã xóa {count} files/thư mục trong: {path}")

if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    synthetic_train = config["paths"]["synthetic_train"]
    synthetic_val = config["paths"]["synthetic_val"]
    
    print(f"CẢNH BÁO: Dữ liệu trong 2 thư mục sau sẽ bị xóa:")
    print(f" - {synthetic_train}")
    print(f" - {synthetic_val}")
    confirm = input("Bạn có chắc chắn muốn xóa không? (y/N): ")
    
    if confirm.lower() == 'y':
        clean_dir(synthetic_train)
        clean_dir(synthetic_val)
        print("Hoàn tất!")
    else:
        print("Đã hủy thao tác.")
