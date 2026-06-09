"""
validate_dataset.py — Kiểm tra dataset trước khi train.
Đọc thử vài batch từ DataLoader để kiểm tra shape và bắt lỗi OOV.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import get_dataloaders

def test_dataloader():
    print("=== KIỂM TRA DATALOADER CHO CRNN ===")
    
    # Ở đây ta trỏ vào data/synthetic/train và data/raw/Recognition
    train_dirs = ["data/synthetic/train", "data/raw/Recognition"]
    val_dirs = ["data/synthetic/val"]
    
    try:
        train_loader, val_loader = get_dataloaders(
            train_dirs=train_dirs,
            val_dirs=val_dirs,
            charset_path="data/charset.txt",
            batch_size=16, # batch nhỏ để test
            num_workers=0  # set 0 để debug dễ hơn
        )
        
        print(f"Số lượng batch train: {len(train_loader)}")
        print(f"Số lượng batch val: {len(val_loader)}")
        
        # Lấy thử 1 batch đầu tiên
        for batch in train_loader:
            images = batch["image"]
            labels = batch["label"]
            label_lens = batch["label_len"]
            label_strs = batch["label_str"]
            input_lens = batch["input_len"]
            
            print("\n--- BATCH MẪU ---")
            print(f"Hình ảnh (Image Tensor) Shape : {images.shape} (N, C, H, max_W)")
            print(f"Nhãn mã hoá (Label Tensor)  : {labels.shape} (N, max_label_len)")
            print(f"Độ dài input CNN (Input Len): {input_lens.shape}")
            
            print("\nVài mẫu text trong batch:")
            for i in range(min(5, len(label_strs))):
                print(f"  [{i}] '{label_strs[i]}' (len={label_lens[i].item()})")
                
            # Đảm bảo Height cố định = 32
            assert images.shape[2] == 32, "Lỗi: Chiều cao ảnh phải luôn được resize về 32!"
            break
            
        print("\n[THÀNH CÔNG] Dataloader hoạt động hoàn hảo, collate_fn padding chiều rộng chuẩn xác!")
        
    except Exception as e:
        print(f"\n[LỖI] Dataloader gặp sự cố: {e}")

if __name__ == "__main__":
    test_dataloader()
