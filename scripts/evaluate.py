"""
evaluate.py — Đánh giá model trên val/test set.
"""
import sys
import os
import yaml
import torch
import jiwer
import random

# Ép Windows không sập
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from src.recognizer.model import CRNN, ctc_greedy_decode
from src.dataset.loader import OCRDataset, collate_fn
from torch.utils.data import DataLoader
from src.trainer.train_cer import CharsetCodec

def main():
    print("1. Khởi tạo cấu hình và DataLoader...")
    CONFIG_PATH = 'configs/default.yaml'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Dùng GPU/CPU: {device}")

    codec = CharsetCodec(config['paths']['charset'])
    val_ds = OCRDataset([config['paths']['synthetic_val']], config['paths']['charset'], is_train=False, target_h=config['preprocess']['target_h'])
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Khởi tạo model base
    model = CRNN(
        num_classes=len(codec),
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        lstm_dropout=0.0
    ).to(device)

    # Danh sách Checkpoint cần test
    checkpoints = {
        'F1 (Phase 1)': 'checkpoints/crnn_best_f1.pth',
        'F2 (Phase 2)': 'checkpoints/crnn_best_f2.pth',
        'F3 (Phase 3)': 'checkpoints/crnn_best_f3.pth'
    }

    # Lưu lại một vài mẫu để in đối chiếu
    sample_targets = []
    sample_preds = {name: [] for name in checkpoints.keys()}

    print("\n2. Bắt đầu đánh giá từng Checkpoint:")
    print("-" * 50)
    
    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"[{name}] Không tìm thấy file {path}")
            continue
            
        # Nạp trọng số
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        
        valid_targets = []
        valid_preds = []
        
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['label_str']
            
            with torch.no_grad():
                out = model(images)
                preds = ctc_greedy_decode(out, codec.charset)
                
            for t, p in zip(targets, preds):
                if len(t.strip()) > 0:
                    valid_targets.append(t)
                    valid_preds.append(p if len(p.strip()) > 0 else ' ')
                    
                    # Lưu lại 5 mẫu ngẫu nhiên (chỉ lưu cho F1 để đồng nhất vị trí)
                    if name == 'F1 (Phase 1)' and random.random() < 0.05 and len(sample_targets) < 5:
                        sample_targets.append(t)
                    
                    # Nếu Target có trong sample_targets thì lưu luôn cả preds
                    if t in sample_targets:
                        sample_preds[name].append(p)
        
        # Tính CER
        cer = jiwer.cer(valid_targets, valid_preds) * 100
        print(f"[{name}] CER = {cer:.2f}%")
        
    print("-" * 50)
    print("\n3. ĐỐI CHIẾU MỘT SỐ MẪU DỰ ĐOÁN:")
    for i, target in enumerate(sample_targets):
        print(f"\n[Ảnh {i+1}] THỰC TẾ: {target}")
        for name in checkpoints.keys():
            if i < len(sample_preds[name]):
                print(f"         {name} : {sample_preds[name][i]}")

if __name__ == "__main__":
    main()
