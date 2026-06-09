import cv2
import numpy as np

def preprocess_image(image: np.ndarray, config: dict, max_size: int = 960) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Tiền xử lý ảnh:
    1. Resize để có chiều rộng/cao là bội số của 32.
    2. Normalize theo ImageNet (mean, std).
    3. Đổi dạng NCHW.
    """
    h, w = image.shape[:2]
    
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        
    new_h = int(np.round(h * scale / 32) * 32)
    new_w = int(np.round(w * scale / 32) * 32)
    
    new_h = max(32, new_h)
    new_w = max(32, new_w)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Normalize
    c_mean = config.get("normalize_mean", [0.485, 0.456, 0.406])
    c_std = config.get("normalize_std", [0.229, 0.224, 0.225])
    mean = np.array(c_mean, dtype=np.float32)
    std = np.array(c_std, dtype=np.float32)
    
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
    
    # NCHW
    img_tensor = img_norm.transpose(2, 0, 1)
    img_tensor = np.expand_dims(img_tensor, axis=0) # (1, 3, H, W)
    
    return img_tensor, (h, w)
