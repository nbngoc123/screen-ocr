"""
corpus.py — Text corpus cho synthetic data generation.

Cung cấp các chuỗi text từ Wikipedia/ICDAR để render lên ảnh synthetic.
"""
from __future__ import annotations

import os
import random
import string

# ─── Load Wiki & ICDAR Corpus ────────────────────────────────────────────────
ICDAR_CORPUS = []
for file_path in ["data/wiki_corpus.txt", "data/icdar_corpus.txt", "data/icdar_en_corpus.txt"]:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if len(line.strip()) >= 2]
                ICDAR_CORPUS.extend(lines)
            # Bỏ print để không bị spam khi chạy multi-worker
        except Exception as e:
            pass
            
if ICDAR_CORPUS:
    pass

# ─── Hard Samples (Cho UI Windows) ────────────────────────────────────────────
HARD_TEXTS = [
    "Il1I|!",
    "O0Oo",
    "rnmvv",
    "VVW",
    "S5s",
    "B8",
    "2Zz",
    "gq9",
    "lIlIlI",
    "OO00OO",
    "HP:100",
    "LV.99",
    "+99999",
]

# ─── Domain-Specific Corpus (UI Windows) ──────────────────────────────────────
UI_SPECIFIC = [
    # Số & đơn vị
    "₫1.234.567", "$99.99", "€1,234.00", "100%", "3.14 GB",
    "08:30 – 17:00", "2024-06-09", "09/06/2024", "T+2",
    # Trạng thái UI
    "Loading… (87%)", "Retry (3/5)", "Page 1 of 24",
    "Step 2 of 4", "Updated 2 min ago", "Syncing…",
    # Path / URL
    r"C:\Users\Admin\Documents\report.pdf",
    "https://example.com/api/v2/users?page=1",
    "\\\\SERVER\\Share\\folder",
    # Code trên screen
    "git commit -m 'fix: null check'",
    "npm run build --prod",
    "pip install torch --index-url ...",
    "SELECT * FROM users WHERE id = 42",
    # VN-specific
    "Mã số thuế: 0123456789",
    "CMND: 079 123 456 789",
    "SĐT: 0912.345.678",
    "Tỉnh/TP: Hà Nội",
]

MIXED_SCRIPT = [
    # EN+VN mixed — rất phổ biến trên Windows VN
    "File đã lưu thành công",
    "Upload ảnh (tối đa 5MB)",
    "Password phải có ít nhất 8 ký tự",
    "Confirm xoá item này?",
]

def random_text(min_len: int = 2, max_len: int = 40, corpus_prob: float = 0.8, word_prob: float = 0.3, phrase_prob: float = 0.4, hard_prob: float = 0.10) -> str:
    """
    Sinh text ngẫu nhiên từ corpus hoặc generate random string.

    Args:
        min_len: Độ dài tối thiểu.
        max_len: Độ dài tối đa (text bị truncate nếu dài hơn).

    Returns:
        Chuỗi text, đã strip và truncate.
    """
    r = random.random()
    
    # Ưu tiên lấy từ danh sách hard samples (dễ nhầm)
    if r < hard_prob:
        return random.choice(HARD_TEXTS)
        
    # Cho thêm 10% cơ hội rơi vào UI_SPECIFIC hoặc MIXED_SCRIPT
    # Do hàm này được thiết kế dựa trên xác suất tích luỹ
    r2 = random.random()
    if r2 < 0.10:
        if random.random() < 0.7:
            return random.choice(UI_SPECIFIC)
        else:
            return random.choice(MIXED_SCRIPT)
    
    # Tính lại xác suất lấy từ corpus
    r3 = random.random()
    
    # Lấy từ siêu corpus Wiki/ICDAR (nếu có)
    if ICDAR_CORPUS and r3 < corpus_prob:
        text = random.choice(ICDAR_CORPUS)
        
        # Băm nhỏ câu theo level: Word (1-2 từ), Phrase (3-5 từ), hoặc Full Sentence
        level = random.random()
        words = text.split()
        if len(words) > 1:
            if level < word_prob:
                # mức Word (1-2 từ)
                num_words = random.randint(1, 2)
            elif level < word_prob + phrase_prob:
                # mức Phrase (3-5 từ)
                num_words = random.randint(3, 5)
            else:
                # mức Full Sentence
                num_words = len(words)
                
            # Lấy một đoạn con liên tiếp ngẫu nhiên
            num_words = min(num_words, len(words))
            start_idx = random.randint(0, len(words) - num_words)
            text = " ".join(words[start_idx : start_idx + num_words])
    # 20% sinh chữ/số ngẫu nhiên
    else:
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

if not ICDAR_CORPUS:
    import warnings
    warnings.warn(
        "ICDAR_CORPUS rỗng — toàn bộ text sẽ là random string. "
        "Thêm file vào data/wiki_corpus.txt để cải thiện chất lượng.",
        stacklevel=1,
    )
