"""
corpus.py — Text corpus cho synthetic data generation.

Cung cấp các chuỗi text theo domain (UI labels, câu tiếng Việt/Anh,
code snippets, số) để render lên ảnh synthetic.
"""
from __future__ import annotations

import random
import string

# ─── Domain corpus ────────────────────────────────────────────────────────────
DOMAINS: dict[str, list[str]] = {
    "ui_labels": [
        "OK", "Cancel", "Apply", "Close", "Save", "Open", "Delete",
        "Settings", "Help", "About", "File", "Edit", "View", "Tools",
        "Cài đặt", "Đóng", "Lưu", "Mở file", "Xoá", "Thoát",
        "Đăng nhập", "Đăng ký", "Xác nhận", "Hủy bỏ", "Tiếp tục"
    ],
    "sentences_vn": [
        "Xin chào người dùng", "Tổng cộng: 1.234.567 VND",
        "Ngày tạo: 09/06/2024", "Trạng thái: Đang xử lý",
        "Mã đơn hàng: ORD-00421", "Họ tên: Nguyễn Văn A",
        "Email: example@gmail.com", "Điện thoại: 0912 345 678",
        "Vui lòng kiểm tra lại thông tin", "Dữ liệu đã được lưu thành công",
        "Lỗi kết nối máy chủ", "Không tìm thấy kết quả phù hợp"
    ],
    "sentences_en": [
        "Welcome to the system", "Total: $1,234.56",
        "Status: Processing", "Order ID: ORD-00421",
        "Please enter your password", "File not found",
        "Connection established", "Loading... please wait",
        "Are you sure you want to delete this?", "Changes saved successfully",
        "Invalid username or password", "No matching results found"
    ],
    "code_snippets": [
        "if __name__ == '__main__':", "import numpy as np",
        "def forward(self, x):", "return torch.sigmoid(x)",
        "print(f'Loss: {loss:.4f}')", "model.train()",
        "for i in range(10):", "class MyNet(nn.Module):",
        "return [x for x in lst if x > 0]"
    ],
    "numbers": [
        "1,234,567", "3.14159", "0x1A2F", "100%",
        "08:30:00", "2024-06-09", "v1.2.3", "#FF5733",
        "192.168.1.1", "50.5 GB", "3,000 đ", "12/12/2024"
    ],
}


def random_text(min_len: int = 2, max_len: int = 40) -> str:
    """
    Sinh text ngẫu nhiên từ corpus hoặc generate tổng hợp.

    Xác suất 70% lấy từ corpus, 30% generate ngẫu nhiên.

    Args:
        min_len: Độ dài tối thiểu.
        max_len: Độ dài tối đa (text bị truncate nếu dài hơn).

    Returns:
        Chuỗi text, đã strip và truncate.
    """
    if random.random() < 0.7:
        # Lấy từ corpus
        domain = random.choice(list(DOMAINS.keys()))
        text = random.choice(DOMAINS[domain])
    else:
        # Generate ngẫu nhiên các chuỗi ngẫu nhiên
        length = random.randint(min_len, max_len)
        chars = string.ascii_letters + string.digits + " "
        text = "".join(random.choices(chars, k=length)).strip()

    # Truncate nếu quá dài
    if len(text) > max_len:
        text = text[:max_len]
        
    # Bảo vệ chống chuỗi rỗng
    if not text:
        text = "OCR"
        
    return text
