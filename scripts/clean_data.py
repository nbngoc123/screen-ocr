"""
Script dọn dẹp dữ liệu (xóa rác, làm sạch tập huấn luyện)
"""
import shutil
from pathlib import Path

def clean_directory(dir_path: str):
    p = Path(dir_path)
    if not p.exists():
        print(f"[Bỏ qua] Không tìm thấy thư mục: {dir_path}")
        return

    count_deleted = 0
    for item in p.rglob("*"):
        # Không xóa file .gitkeep để giữ cấu trúc thư mục
        if item.is_file() and item.name != ".gitkeep":
            try:
                item.unlink()
                count_deleted += 1
            except Exception as e:
                print(f"Lỗi khi xóa {item}: {e}")
                
    print(f"[Thành công] Đã xóa {count_deleted} file trong thư mục: {dir_path}")

def main():
    print("=== CÔNG CỤ DỌN DẸP DỮ LIỆU ===")
    print("1. Xóa toàn bộ dữ liệu ICDAR (data/icdar)")
    print("2. Xóa toàn bộ dữ liệu giả (Synthetic - data/synthetic)")
    print("3. Xóa CẢ HAI")
    print("0. Hủy bỏ")
    
    choice = input("\nNhập lựa chọn của bạn (0-3): ").strip()
    
    if choice == "1":
        clean_directory("data/icdar")
    elif choice == "2":
        clean_directory("data/synthetic/train")
        clean_directory("data/synthetic/val")
    elif choice == "3":
        clean_directory("data/icdar")
        clean_directory("data/synthetic/train")
        clean_directory("data/synthetic/val")
    elif choice == "0":
        print("Đã hủy thao tác.")
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
