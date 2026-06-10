"""
Script: tests/test_train.py
Vừa train vừa test trực tiếp độ chính xác trên hình ảnh test đời thực (D:\test2.png).
"""
import argparse
import logging
import os
import sys
import time
import yaml
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import jiwer

from src.dataset.loader import get_dataloaders
from src.dataset.charset import CharsetCodec
from src.recognizer.model import CRNN, ctc_greedy_decode
from src.data_generation.augment import preprocess_for_crnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate(model, val_loader, criterion, codec, device):
    model.eval()
    losses = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            images, labels, label_len = batch["image"].to(device), batch["label"].to(device), batch["label_len"].to(device)
            outputs = model(images)
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)
            loss = criterion(log_probs, labels.cpu(), input_lengths, label_len.cpu())
            losses.append(loss.item())
            preds = ctc_greedy_decode(outputs, codec.charset)
            all_preds.extend(preds)
            all_targets.extend(batch["label_str"])
            
    val_loss = sum(losses) / len(losses) if losses else 0.0
    valid_preds, valid_targets = [], []
    for p, t in zip(all_preds, all_targets):
        if len(t.strip()) > 0:
            valid_preds.append(p if len(p.strip()) > 0 else " ")
            valid_targets.append(t)
    try:
        val_cer = jiwer.cer(valid_targets, valid_preds)
    except:
        val_cer = 1.0
    return val_loss, val_cer

def test_real_image(model, codec, device, image_path, target_h):
    """Cắt vài ô từ ảnh test để theo dõi độ học hỏi đời thực"""
    if not os.path.exists(image_path):
        return
        
    img = cv2.imread(image_path)
    if img is None:
        return
        
    # Một số toạ độ cắt từ bảng của D:\test2.png
    crops = [
        {"name": "Khóa chính (Primary Key)", "coords": (3184, 3265, 1173, 2187)},
        {"name": "filename", "coords": (2580, 2666, 1173, 2103)},
        {"name": "int", "coords": (2780, 2876, 89, 202)}
    ]
    
    model.eval()
    with torch.no_grad():
        logger.info("   [🔍] KẾT QUẢ ĐỜI THỰC:")
        for crop_info in crops:
            y1, y2, x1, x2 = crop_info["coords"]
            crop_img = img[y1:y2, x1:x2]
            
            # Preprocess như trong Inference
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
            logger.info(f"        - {crop_info['name']}: '{text}'")

def main():
    parser = argparse.ArgumentParser(description="Vừa train vừa test trực quan với dữ liệu đời thực")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--test-image", type=str, default="D:\\test2.png", help="Đường dẫn ảnh test để crop")
    parser.add_argument("--epochs", type=int, default=100, help="Số lượng epoch cần train")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch size trong config")
    parser.add_argument("--train-samples", type=int, default=2000, help="Số ảnh train để debug (-1 để dùng toàn bộ)")
    parser.add_argument("--val-samples", type=int, default=500, help="Số ảnh val để debug (-1 để dùng toàn bộ)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    target_h = config["preprocess"]["target_h"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    lr = config["training"]["lr"]
    epochs = args.epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Khởi chạy huấn luyện test trên thiết bị: {device}")
    
    logger.info("Đang load dữ liệu...")
    train_loader, val_loader = get_dataloaders(
        train_dirs=[config["paths"]["synthetic_train"]],
        val_dirs=[config["paths"]["synthetic_val"]],
        charset_path=config["paths"]["charset"],
        batch_size=batch_size,
        num_workers=0, # Set 0 for Windows debugging
        target_h=target_h
    )
    
    # Cắt bộ data nhỏ lại để debug
    if args.train_samples > 0:
        logger.info(f"Cắt bộ data train xuống còn {args.train_samples} mẫu...")
        train_loader.dataset.samples = train_loader.dataset.samples[:args.train_samples]
    if args.val_samples > 0:
        val_loader.dataset.samples = val_loader.dataset.samples[:args.val_samples]

    codec = CharsetCodec(config["paths"]["charset"])
    model = CRNN(
        num_classes=len(codec),
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=config["model"]["lstm_dropout"]
    ).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        batch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            images, labels, label_len = batch["image"].to(device), batch["label"].to(device), batch["label_len"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)
            loss = criterion(log_probs, labels.cpu(), input_lengths, label_len.cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        val_loss, val_cer = evaluate(model, val_loader, criterion, codec, device)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {time.time()-start_time:.1f}s")
        
        # In trực tiếp kết quả cắt từ ảnh test sau mỗi Epoch
        test_real_image(model, codec, device, args.test_image, target_h)

if __name__ == "__main__":
    main()
