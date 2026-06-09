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
            print(f"Đã load {len(lines)} câu từ {file_path}")
        except Exception as e:
            print(f"Lỗi load corpus {file_path}: {e}")
            
if ICDAR_CORPUS:
    print(f"-> Tổng cộng có {len(ICDAR_CORPUS)} câu trong siêu Corpus!")


def random_text(min_len: int = 2, max_len: int = 40, corpus_prob: float = 0.8, word_prob: float = 0.3, phrase_prob: float = 0.4) -> str:
    """
    Sinh text ngẫu nhiên từ corpus hoặc generate random string.

    Args:
        min_len: Độ dài tối thiểu.
        max_len: Độ dài tối đa (text bị truncate nếu dài hơn).

    Returns:
        Chuỗi text, đã strip và truncate.
    """
    r = random.random()
    
    # Lấy từ siêu corpus Wiki/ICDAR (nếu có)
    if ICDAR_CORPUS and r < corpus_prob:
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
