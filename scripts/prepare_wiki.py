"""
Script tải và xử lý Wikipedia Tiếng Việt từ Hugging Face để làm Corpus sinh ảnh.
"""
import re
import os
from datasets import load_dataset

def prepare_wiki_corpus(output_file="data/wiki_corpus.txt", target_sentences=500000):
    print("Đang kết nối tới Hugging Face để tải dataset DataStudio/Viet-wikipedia...")
    print("Vì dùng chế độ streaming=True nên sẽ không tốn dung lượng ổ cứng tải cả bộ!")
    
    ds = load_dataset("DataStudio/Viet-wikipedia", split="train", streaming=True)
    
    corpus = set()
    count = 0
    
    print(f"Đang bóc tách lấy {target_sentences} câu (độ dài 5-80 ký tự)...")
    for item in ds:
        text = item.get('text', '')
        # Tách text thành các câu nhỏ dựa trên dấu câu và xuống dòng
        chunks = re.split(r'[.;!?"\n]+', text)
        
        for chunk in chunks:
            chunk = chunk.strip()
            # Có thể cắt thêm theo dấu phẩy nếu đoạn quá dài
            if len(chunk) > 80:
                sub_chunks = chunk.split(',')
                for sc in sub_chunks:
                    sc = sc.strip()
                    if 5 <= len(sc) <= 80:
                        corpus.add(sc)
            else:
                if 5 <= len(chunk) <= 80:
                    corpus.add(chunk)
                    
        # Update tiến độ
        if len(corpus) - count >= 10000:
            count = len(corpus)
            print(f" Đã thu thập: {count:,} câu")
            
        if len(corpus) >= target_sentences:
            break
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nĐang lưu {len(corpus):,} câu vào {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line + "\n")
            
    print("Hoàn tất! Bộ sinh ảnh Synthetic đã sẵn sàng bùng nổ Tiếng Việt!")

if __name__ == "__main__":
    prepare_wiki_corpus()
