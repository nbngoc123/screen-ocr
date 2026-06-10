import os
import sys
import yaml
from pathlib import Path
import multiprocessing
import shutil
import random
from collections import defaultdict

# Thêm thư mục gốc vào đường dẫn hệ thống để Python tìm thấy module src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import generate_dataset

def get_font_weights(fonts: list[str], common_fonts: list[str], common_wt: float, rare_wt: float) -> list[float]:
    """Tạo mảng xác suất: Font phổ biến (thẳng) x trọng số cao, font lạ x trọng số thấp."""
    weights = []
    for f in fonts:
        f_lower = f.lower()
        if any(cf in f_lower for cf in common_fonts):
            weights.append(common_wt)
        else:
            weights.append(rare_wt)
    return weights

def check_output_dir(output_dir: str) -> bool:
    """Trả về False nếu người dùng từ chối ghi đè dữ liệu, True nếu có thể tiếp tục."""
    p = Path(output_dir)
    if not p.exists():
        return True
    
    existing_imgs = len(list(p.glob("*.png")))
    if existing_imgs > 0:
        print(f"[WARN] {output_dir} đã có {existing_imgs} ảnh.")
        ans = input("Xoá và sinh lại? [y/N]: ").strip().lower()
        if ans == "y":
            shutil.rmtree(output_dir)
            p.mkdir(parents=True)
            print("Đã xoá.")
            return True
        else:
            print("Bỏ qua thư mục này.")
            return False
    return True

def worker_generate(args):
    """Worker function cho multiprocessing"""
    (font_list, output_dir, n_samples, font_weights, worker_id, 
     corpus_prob, word_split_prob, phrase_split_prob,
     hard_prob, shadow_prob, stroke_prob, bg_light_prob, bg_dark_prob,
     real_bg_prob, neg_prob, multi_dpi_prob) = args
    
    generate_dataset(
        font_list, output_dir, n_samples, 
        font_weights=font_weights, worker_id=worker_id, 
        corpus_prob=corpus_prob, word_split_prob=word_split_prob, phrase_split_prob=phrase_split_prob,
        hard_prob=hard_prob, shadow_prob=shadow_prob, stroke_prob=stroke_prob,
        bg_light_prob=bg_light_prob, bg_dark_prob=bg_dark_prob,
        real_bg_prob=real_bg_prob, 
        neg_prob=neg_prob, multi_dpi_prob=multi_dpi_prob
    )

def run_multiprocess(
    font_list, output_dir, total_samples, font_weights, n_workers=4, 
    corpus_prob=0.8, word_split_prob=0.3, phrase_split_prob=0.4,
    hard_prob=0.10, shadow_prob=0.25, stroke_prob=0.30,
    bg_light_prob=0.60, bg_dark_prob=0.20,
    real_bg_prob=0.30, neg_prob=0.05, multi_dpi_prob=0.20
):
    """Chia nhỏ tổng số sample ra cho các worker chạy song song."""
    if n_workers <= 1:
        generate_dataset(
            font_list, output_dir, total_samples, 
            font_weights=font_weights, worker_id=0, 
            corpus_prob=corpus_prob, word_split_prob=word_split_prob, phrase_split_prob=phrase_split_prob,
            hard_prob=hard_prob, shadow_prob=shadow_prob, stroke_prob=stroke_prob,
            bg_light_prob=bg_light_prob, bg_dark_prob=bg_dark_prob,
            real_bg_prob=real_bg_prob,
            neg_prob=neg_prob, multi_dpi_prob=multi_dpi_prob
        )
        # Rename file nhãn
        lbl_file = Path(output_dir) / "labels_0.jsonl"
        if lbl_file.exists():
            lbl_file.rename(Path(output_dir) / "labels.jsonl")
        print(f"Generated {total_samples} samples -> {output_dir}")
        return

    samples_per_worker = total_samples // n_workers
    remainder = total_samples % n_workers
    args_list = []
    
    for i in range(n_workers):
        n = samples_per_worker + (1 if i < remainder else 0)
        args_list.append((
            font_list, output_dir, n, font_weights, i, 
            corpus_prob, word_split_prob, phrase_split_prob,
            hard_prob, shadow_prob, stroke_prob, bg_light_prob, bg_dark_prob,
            real_bg_prob, neg_prob, multi_dpi_prob
        ))
        
    print(f"Khởi động {n_workers} workers...")
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(worker_generate, args_list)
        
    # Gom tất cả file labels của các worker thành 1 file labels.jsonl duy nhất
    out_path = Path(output_dir)
    merged_labels = []
    for i in range(n_workers):
        lbl_file = out_path / f"labels_{i}.jsonl"
        if lbl_file.exists():
            merged_labels.extend(lbl_file.read_text(encoding='utf-8').splitlines())
            lbl_file.unlink() # Xóa file tạm
            
    with open(out_path / "labels.jsonl", "w", encoding='utf-8') as f:
        if merged_labels:
            f.write("\n".join(merged_labels) + "\n")
            
    print(f"\n[Hoàn tất] Generated {total_samples} samples -> {output_dir}")

if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    try:
        fonts = Path(config["paths"]["font_list"]).read_text(encoding="utf-8").splitlines()
        fonts = [f.strip() for f in fonts if f.strip()]
        if not fonts:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Chưa có danh sách font. Chạy scripts/list_fonts.py trước.")
        exit(1)
        
    cfg_dg = config["data_gen"]
    common_fonts = cfg_dg.get("common_fonts", [])
    common_wt = cfg_dg.get("common_font_weight", 10.0)
    rare_wt = cfg_dg.get("rare_font_weight", 1.0)
    val_ratio = cfg_dg.get("val_ratio", 0.1)
    
    corpus_prob = cfg_dg.get("corpus_prob", 0.8)
    word_split_prob = cfg_dg.get("word_split_prob", 0.3)
    phrase_split_prob = cfg_dg.get("phrase_split_prob", 0.4)
    
    # UI/Hard parameters
    hard_prob = cfg_dg.get("hard_prob", 0.10)
    shadow_prob = cfg_dg.get("shadow_prob", 0.25)
    stroke_prob = cfg_dg.get("stroke_prob", 0.30)
    bg_light_prob = cfg_dg.get("bg_light_prob", 0.60)
    bg_dark_prob = cfg_dg.get("bg_dark_prob", 0.20)
    real_bg_prob = cfg_dg.get("real_bg_prob", 0.30)
    neg_prob = cfg_dg.get("neg_prob", 0.05)
    multi_dpi_prob = cfg_dg.get("multi_dpi_prob", 0.20)

    # Split train and val fonts theo nguyên tắc không trùng lặp (group by family)
    font_families = defaultdict(list)
    for f in fonts:
        # Tách tên họ font (VD: "Arial Bold.ttf" -> "arial")
        base = os.path.basename(f)
        family = base.replace('.ttf', '').replace('.otf', '').split('-')[0].split(' ')[0].split('_')[0].lower()
        font_families[family].append(f)
        
    families = list(font_families.keys())
    random.seed(42) # Giữ cho việc split luôn cố định qua các lần sinh
    random.shuffle(families)
    
    split_idx = int(len(families) * (1.0 - val_ratio))
    train_families = families[:split_idx]
    val_families = families[split_idx:]
    
    train_fonts = []
    for fam in train_families:
        train_fonts.extend(font_families[fam])
        
    val_fonts = []
    for fam in val_families:
        val_fonts.extend(font_families[fam])
    
    # Số lượng ảnh lấy thẳng từ file cấu hình default.yaml
    n_train = cfg_dg["n_train"]
    n_val = cfg_dg["n_val"]
    
    n_workers = cfg_dg.get("n_workers", 4)
    
    print(f"Sử dụng {len(train_fonts)} fonts cho tập Train, {len(val_fonts)} fonts cho tập Val.")
    
    train_dir = config["paths"]["synthetic_train"]
    val_dir = config["paths"]["synthetic_val"]
    
    do_train = check_output_dir(train_dir)
    do_val = check_output_dir(val_dir)
    
    if not do_train and not do_val:
        print("Đã bỏ qua cả Train và Val. Thoát.")
        exit(0)
    
    print(f"Bắt đầu sinh dữ liệu ({n_train} Train, {n_val} Val) với {n_workers} processes...")
    
    # Tính toán trọng số xác suất cho các font
    train_weights = get_font_weights(train_fonts, common_fonts, common_wt, rare_wt)
    val_weights = get_font_weights(val_fonts, common_fonts, common_wt, rare_wt)
    
    # Train
    if do_train:
        print("\n--- Tập Train ---")
        run_multiprocess(
            font_list=train_fonts,
            output_dir=train_dir,
            total_samples=n_train,
            font_weights=train_weights,
            n_workers=n_workers,
            corpus_prob=corpus_prob,
            word_split_prob=word_split_prob,
            phrase_split_prob=phrase_split_prob,
            hard_prob=hard_prob,
            shadow_prob=shadow_prob,
            stroke_prob=stroke_prob,
            bg_light_prob=bg_light_prob,
            bg_dark_prob=bg_dark_prob,
            real_bg_prob=real_bg_prob,
            neg_prob=neg_prob,
            multi_dpi_prob=multi_dpi_prob
        )
    
    # Val
    if do_val:
        print("\n--- Tập Validation ---")
        run_multiprocess(
            font_list=val_fonts,
            output_dir=val_dir,
            total_samples=n_val,
            font_weights=val_weights,
            n_workers=n_workers,
            corpus_prob=corpus_prob,
            word_split_prob=word_split_prob,
            phrase_split_prob=phrase_split_prob,
            hard_prob=hard_prob,
            shadow_prob=shadow_prob,
            stroke_prob=stroke_prob,
            bg_light_prob=bg_light_prob,
            bg_dark_prob=bg_dark_prob,
            real_bg_prob=real_bg_prob,
            neg_prob=neg_prob,
            multi_dpi_prob=multi_dpi_prob
        )
    
    print("Hoàn tất!")
