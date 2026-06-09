"""
run_gen.py — Chạy data generator.
"""
import yaml
from pathlib import Path

from src.data_gen import generate_dataset

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
        
    # TODO: implement logic dùng multiprocessing để gen data
    # generate_dataset(...)
    print("Đang phát triển: logic generate_dataset()")
