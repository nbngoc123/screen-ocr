"""
list_fonts.py — Liệt kê toàn bộ system fonts.
"""
import glob
from pathlib import Path

FONT_DIRS = [
    r"C:\Windows\Fonts",
    # r"C:\Users\{username}\AppData\Local\Microsoft\Windows\Fonts",
]

def get_all_fonts() -> list[str]:
    fonts = []
    for d in FONT_DIRS:
        fonts.extend(glob.glob(f"{d}\\*.ttf"))
        fonts.extend(glob.glob(f"{d}\\*.otf"))
    
    # Lọc bỏ các biến thể bold, italic...
    filtered = []
    for f in fonts:
        name = Path(f).stem.lower()
        if any(s in name for s in ["bold", "italic", "heavy", "black", "light", "thin"]):
            continue
        filtered.append(f)
    print(f"Tìm thấy {len(filtered)} regular fonts")
    return filtered

if __name__ == "__main__":
    fonts = get_all_fonts()
    Path("data").mkdir(exist_ok=True)
    Path("data/font_list.txt").write_text("\n".join(fonts), encoding="utf-8")
    print("Đã lưu vào data/font_list.txt")
