"""
Script hỗ trợ tải trọng số mô hình từ Google Drive (Dùng cho Colab hoặc Client).
Yêu cầu cài đặt: pip install gdown
"""
import os
import argparse
from pathlib import Path

try:
    import gdown
except ImportError:
    print("[!] Vui lòng cài đặt thư viện gdown: pip install gdown")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Tải Checkpoint từ Google Drive")
    parser.add_argument("--file-id", type=str, help="Google Drive File ID của file crnn_best.pth", required=True)
    parser.add_argument("--output", type=str, default="checkpoints/crnn_best.pth", help="Đường dẫn lưu file")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Đang tải trọng số từ Google Drive (File ID: {args.file_id})...")
    url = f"https://drive.google.com/uc?id={args.file_id}"
    
    gdown.download(url, str(out_path), quiet=False)
    
    if out_path.exists():
        print(f"\n[THÀNH CÔNG] Đã tải xong trọng số và lưu tại: {out_path}")
    else:
        print("\n[LỖI] Tải thất bại. Vui lòng kiểm tra lại File ID hoặc quyền truy cập link Drive.")

if __name__ == "__main__":
    main()
