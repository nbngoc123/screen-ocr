"""
Script chuyển đổi gt.txt của tập ICDAR Recognition thành labels.jsonl
và trích xuất text vào file icdar_en_corpus.txt
"""
import json
import os
from pathlib import Path

def convert_icdar_recognition(gt_path="data/raw/Recognition/gt.txt", 
                              out_jsonl="data/raw/Recognition/labels.jsonl",
                              out_corpus="data/icdar_en_corpus.txt"):
    
    gt_file = Path(gt_path)
    if not gt_file.exists():
        print(f"Lỗi: Không tìm thấy {gt_path}")
        return

    corpus_texts = set()
    jsonl_lines = []
    
    with open(gt_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(', ', 1)
            if len(parts) == 2:
                img_file = parts[0]
                label = parts[1].strip('"')  # Loại bỏ dấu ngoặc kép 2 đầu
                
                # Bỏ qua các nhãn rỗng
                if not label:
                    continue
                    
                # Dữ liệu cho jsonl
                jsonl_lines.append(json.dumps({"file": img_file, "label": label}, ensure_ascii=False))
                
                # Dữ liệu cho corpus (chỉ lấy chữ cái, bỏ ký tự đặc biệt nếu muốn, ở đây giữ nguyên)
                if len(label) >= 2:
                    corpus_texts.add(label)

    # Lưu file labels.jsonl
    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")
    print(f"Success: Created {out_jsonl} with {len(jsonl_lines)} images.")
    
    # Lưu file corpus
    with open(out_corpus, "w", encoding="utf-8") as f:
        for text in corpus_texts:
            f.write(text + "\n")
    print(f"Success: Extracted {len(corpus_texts)} English words to {out_corpus}.")

if __name__ == "__main__":
    convert_icdar_recognition()
