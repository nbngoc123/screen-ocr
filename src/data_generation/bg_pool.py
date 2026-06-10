import mss
import cv2
import numpy as np
import random
from pathlib import Path

def build_bg_pool(output_dir: str = "data/bg_pool", n: int = 500):
    """Chụp nhiều screenshot, crop patch ngẫu nhiên làm background pool."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    with mss.mss() as sct:
        # Nếu có nhiều monitor, chọn màn hình chính hoặc phụ tùy hệ thống. Ở đây dùng màn 1
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        
        for i in range(n):
            # Lấy ảnh màn hình
            shot = np.array(sct.grab(monitor))[:, :, :3]
            h, w = shot.shape[:2]
            
            # Kích thước crop ngẫu nhiên
            ph = random.randint(20, 60)
            pw = random.randint(60, 300)
            
            # Tọa độ ngẫu nhiên
            y = random.randint(0, h - ph)
            x = random.randint(0, w - pw)
            
            crop = shot[y:y+ph, x:x+pw]
            
            # Lưu file
            cv2.imwrite(str(out_path / f"{i:04d}.png"), crop)
            
if __name__ == "__main__":
    print("Bắt đầu sinh Background Pool từ screenshot...")
    build_bg_pool()
    print("Hoàn thành!")
