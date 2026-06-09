"""
Script tiền xử lý dữ liệu ICDAR Robust Reading & MLT.
Chức năng:
1. Đọc ảnh và file nhãn toạ độ (gt_*.txt hoặc *.txt).
2. Cắt các vùng chứa chữ và lưu lại dưới định dạng chuẩn của project.
3. Trích xuất toàn bộ text thật gom thành file `data/icdar_corpus.txt` để bộ sinh ảnh giả (Synthetic) học theo.
"""
import os
import cv2
import json
import glob
from pathlib import Path
from tqdm import tqdm

def parse_icdar_line(line: str):
    """
    Parse dòng text của ICDAR.
    Định dạng phổ biến: x1,y1,x2,y2,x3,y3,x4,y4,transcription
    Định dạng MLT: x1,y1,x2,y2,x3,y3,x4,y4,language,transcription
    """
    line = line.strip()
    if not line:
        return None
        
    parts = line.split(',')
    if len(parts) < 9:
        return None
        
    try:
        # Lấy 8 toạ độ
        coords = [int(float(p)) for p in parts[:8]]
    except ValueError:
        return None
        
    rest = ",".join(parts[8:])
    
    # Xử lý trường hợp MLT có chèn language code ở giữa
    known_langs = ['Arabic', 'Latin', 'Chinese', 'Korean', 'Japanese', 'Bangla', 'Hindi', 'Symbols', 'None', 'Mixed', 'Vietnamese', 'French', 'English']
    transcription = rest
    for lang in known_langs:
        if rest.startswith(lang + ","):
            transcription = rest[len(lang)+1:]
            break
            
    # Xử lý nhãn lỗi của ICDAR
    if transcription == "###" or not transcription:
        return None
        
    return coords, transcription

def prepare_icdar_dataset(raw_dir: str, output_dir: str, corpus_file: str):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả file ảnh
    extensions = ['*.jpg', '*.png', '*.jpeg']
    img_files = []
    for ext in extensions:
        img_files.extend(raw_path.rglob(ext))
        
    if not img_files:
        print(f"Không tìm thấy ảnh nào trong {raw_dir}")
        return
        
    print(f"Tìm thấy {len(img_files)} ảnh gốc. Bắt đầu xử lý...")
    
    metadata = []
    corpus_text = set()
    crop_count = 0
    
    for img_path in tqdm(img_files):
        # Tìm file text tương ứng
        # Thường tên là gt_img_name.txt hoặc img_name.txt
        txt_name = img_path.with_suffix('.txt').name
        gt_txt_name = f"gt_{txt_name}"
        
        txt_path = img_path.parent / txt_name
        gt_txt_path = img_path.parent / gt_txt_name
        
        target_txt = None
        if gt_txt_path.exists():
            target_txt = gt_txt_path
        elif txt_path.exists():
            target_txt = txt_path
            
        if not target_txt:
            continue
            
        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Đọc nhãn
        with open(target_txt, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parsed = parse_icdar_line(line)
            if not parsed:
                continue
                
            coords, text = parsed
            
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            if x_max <= x_min or y_max <= y_min:
                continue
                
            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
                
            # Lưu ảnh crop
            crop_name = f"icdar_{crop_count:07d}.png"
            cv2.imwrite(str(out_path / crop_name), crop)
            
            metadata.append({
                "file": crop_name,
                "label": text
            })
            corpus_text.add(text)
            crop_count += 1
            
    # Lưu metadata
    with open(out_path / "labels.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            
    # Cập nhật corpus chung
    corpus_path = Path(corpus_file)
    existing_corpus = set()
    if corpus_path.exists():
        existing_corpus = set(corpus_path.read_text(encoding='utf-8').splitlines())
        
    combined_corpus = existing_corpus.union(corpus_text)
    corpus_path.write_text("\n".join(sorted(list(combined_corpus))), encoding='utf-8')
    
    print(f"\nHoàn tất! Đã crop {crop_count} ảnh lưu vào {output_dir}")
    print(f"Đã cập nhật {len(corpus_text)} câu mới vào {corpus_file} (Tổng: {len(combined_corpus)} câu)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu ICDAR cho CRNN")
    parser.add_argument("--raw_dir", required=True, help="Thư mục chứa ảnh và file .txt gốc của ICDAR")
    parser.add_argument("--output_dir", default="data/icdar", help="Thư mục lưu ảnh đã crop")
    parser.add_argument("--corpus_file", default="data/icdar_corpus.txt", help="File lưu gom text cho Synthetic data")
    
    args = parser.parse_args()
    prepare_icdar_dataset(args.raw_dir, args.output_dir, args.corpus_file)
