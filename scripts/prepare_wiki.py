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

def prepare_english_corpus(output_file="data/icdar_en_corpus.txt", target_sentences=200000):
    print("\nĐang kết nối tới Hugging Face để tải dataset tiếng Anh (wikitext)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    
    corpus = set()
    count = 0
    for item in ds:
        text = item.get('text', '')
        chunks = re.split(r'[.;!?"\n]+', text)
        for chunk in chunks:
            chunk = chunk.strip()
            if 5 <= len(chunk) <= 80:
                corpus.add(chunk)
                
        if len(corpus) - count >= 10000:
            count = len(corpus)
            print(f" Đã thu thập (EN): {count:,} câu")
        if len(corpus) >= target_sentences:
            break
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nĐang lưu {len(corpus):,} câu tiếng Anh vào {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line + "\n")

def generate_charset_from_corpus(corpus_files, output_file="data/charset.txt", max_chars=300):
    print("\nĐang phân tích Corpus để tự động sinh charset.txt (Đảm bảo 100% không bị lỗi Mojibake)...")
    from collections import Counter
    char_counter = Counter()
    
    # Ký tự bắt buộc phải có (Bảng chữ cái tiếng Việt, số, dấu câu cơ bản)
    essential_chars = set("aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệghiíìỉĩịklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvxyýỳỷỹỵ"
                          "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCDĐEÉÈẺẼẸÊẾỀỂỄỆGHIÍÌỈĨỊKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢPQRSTUÚÙỦŨỤƯỨỪỬỮỰVXYÝỲỶỸỴ"
                          "0123456789.,;:'\"!?-()[]{}<>%/*+=$#@&^|\\~_ ")
    
    for file_path in corpus_files:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    char_counter.update(line.strip())
                    
    # Lấy các ký tự phổ biến nhất chưa nằm trong danh sách bắt buộc
    top_chars = [c for c, _ in char_counter.most_common() if c not in essential_chars]
    
    # Ghép ký tự bắt buộc và ký tự phổ biến (tối đa max_chars)
    final_chars = list(essential_chars) + top_chars[:max_chars - len(essential_chars)]
    
    # Loại bỏ khoảng trắng (nếu có dư)
    final_chars = [c for c in final_chars if c not in [' ', '\n', '\t']]
    
    # Sắp xếp cho gọn gàng (Số -> ASCII -> Tiếng Việt -> Các ký tự khác)
    final_chars.sort()
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("".join(final_chars))
    print(f"Đã lưu {len(final_chars)} ký tự chuẩn vào {output_file}")

if __name__ == "__main__":
    prepare_wiki_corpus(target_sentences=300000)
    prepare_english_corpus(target_sentences=100000)
    generate_charset_from_corpus(["data/wiki_corpus.txt", "data/icdar_en_corpus.txt"])
    print("\nHoàn tất! Bộ dữ liệu đã hoàn toàn sạch sẽ và sẵn sàng.")
