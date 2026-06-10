import torch
import cv2
import numpy as np
import yaml
from src.recognizer.model import CRNN
from src.dataset.charset import CharsetCodec
from src.data_generation.augment import preprocess_for_crnn

def test_pytorch():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    codec = CharsetCodec(config["paths"]["charset"])
    model = CRNN(num_classes=len(codec.charset))
    model.load_state_dict(torch.load(config["paths"]["best_weight"], map_location="cpu"))
    model.eval()
    
    img = cv2.imread("D:\\test2.png")
    # Tự crop Box 1: (1173, 3184) -> (2187, 3265) -> "Khóa chính (Primary Key), tự tăng"
    crop = img[3184:3265, 1173:2187]
    
    target_h = config.get("preprocess", {}).get("target_h", 64)
    # Preprocess
    img_tensor_np = preprocess_for_crnn(crop, target_h=target_h, is_train=False)
    img_tensor = torch.from_numpy(img_tensor_np).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        logits = model(img_tensor) # (1, W, num_classes)
        
        # Softmax
        probs_tensor = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs_tensor, dim=-1)[0]
        probs = torch.max(probs_tensor, dim=-1)[0][0]
        
        char_list = []
        conf_list = []
        for i, char_idx in enumerate(preds.tolist()):
            if char_idx != 0 and (not (i > 0 and char_idx == preds[i - 1])):
                char_list.append(codec.charset[char_idx - 1])
                conf_list.append(probs[i].item())
                
        text = "".join(char_list)
        conf = float(np.mean(conf_list)) if conf_list else 0.0
        
        print(f"PyTorch prediction: Text='{text}', Conf={conf:.4f}")

if __name__ == "__main__":
    test_pytorch()
