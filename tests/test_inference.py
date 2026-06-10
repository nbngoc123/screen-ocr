import argparse
import logging
import sys
import yaml
import cv2
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset.charset import CharsetCodec
from src.recognizer.model import CRNN
from src.data_generation.augment import preprocess_for_crnn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def test_real_image(model, codec, device, image_path, target_h):
    if not image_path.exists():
        logger.error(f"Không tìm thấy ảnh: {image_path}")
        return
        
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error("Lỗi đọc ảnh!")
        return
        
    # Các vùng chứa chữ trong D:\test2.png
    crops = [
        {"name": "Khóa chính (Primary Key)", "coords": (3184, 3265, 1173, 2187)},
        {"name": "filename", "coords": (2580, 2666, 1173, 2103)},
        {"name": "int", "coords": (2780, 2876, 89, 202)}
    ]
    
    model.eval()
    with torch.no_grad():
        logger.info(f"--- KẾT QUẢ SUY LUẬN TRÊN ẢNH {image_path.name} ---")
        for crop_info in crops:
            y1, y2, x1, x2 = crop_info["coords"]
            crop_img = img[y1:y2, x1:x2]
            
            img_tensor_np = preprocess_for_crnn(crop_img, target_h=target_h, is_train=False)
            img_tensor = torch.from_numpy(img_tensor_np).unsqueeze(0).unsqueeze(0).float().to(device)
            
            logits = model(img_tensor)
            probs_tensor = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs_tensor, dim=-1)[0]
            
            char_list = []
            for i, char_idx in enumerate(preds.tolist()):
                if char_idx != 0 and (not (i > 0 and char_idx == preds[i - 1])):
                    char_list.append(codec.charset[char_idx - 1])
            text = "".join(char_list)
            logger.info(f"  [+] {crop_info['name']}: '{text}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="checkpoints/crnn.pth")
    parser.add_argument("--image", type=str, default="D:\\test2.png")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    target_h = config["preprocess"]["target_h"]
    codec = CharsetCodec(config["paths"]["charset"])
    
    model = CRNN(
        num_classes=len(codec),
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=0.0
    )
    
    weight_path = Path(args.weight)
    if not weight_path.exists():
        logger.error(f"Chưa có file weight: {weight_path}")
        return
        
    logger.info(f"Đang load mô hình từ {weight_path}...")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    
    test_real_image(model, codec, device, Path(args.image), target_h)

if __name__ == "__main__":
    main()
