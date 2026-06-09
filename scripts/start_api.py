"""
start_api.py - Script hỗ trợ khởi động server API nhanh
"""
import uvicorn
import sys
import os

# Thêm thư mục gốc vào path để import src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("Đang khởi động Screen OCR API Server...")
    # Chạy uvicorn server, trỏ tới đối tượng 'app' trong src/api/app.py
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
