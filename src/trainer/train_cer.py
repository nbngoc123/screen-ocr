"""
train.py — Script huấn luyện mô hình CRNN.

Hỗ trợ debug mode để giới hạn số lượng mẫu huấn luyện, giúp kiểm tra luồng chạy nhanh chóng.
"""
import argparse
import logging
import os
import sys
import time
import yaml
import csv
from pathlib import Path

# Thêm thư mục gốc của project vào PYTHONPATH để import được thư mục src
project_root = Path(__file__).resolve().parent.parent.parent
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate(model, val_loader, criterion, codec, device):
    """Đánh giá mô hình trên tập validation."""
    model.eval()
    losses = []
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            label_len = batch["label_len"].to(device)
            
            outputs = model(images)  # (B, T, C)
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)
            
            loss = criterion(log_probs, labels.cpu(), input_lengths, label_len.cpu())
            losses.append(loss.item())
            
            # Giải mã Prediction để tính CER
            preds = ctc_greedy_decode(outputs, codec.charset)
            targets = batch["label_str"]
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    val_loss = sum(losses) / len(losses) if losses else 0.0
    
    # Tính CER (Bỏ qua các target bị rỗng để tránh lỗi jiwer)
    valid_preds = []
    valid_targets = []
    for p, t in zip(all_preds, all_targets):
        if len(t.strip()) > 0:
            valid_preds.append(p if len(p.strip()) > 0 else " ")
            valid_targets.append(t)
            
    try:
        val_cer = jiwer.cer(valid_targets, valid_preds)
    except Exception as e:
        val_cer = 1.0
        
    return val_loss, val_cer


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình CRNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--train-dir", type=str, default=None, help="Thư mục train data")
    parser.add_argument("--val-dir", type=str, default=None, help="Thư mục val data")
    parser.add_argument("--charset", type=str, default=None, help="File charset")
    parser.add_argument("--batch-size", type=int, default=None, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=None, help="Số lượng epoch")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate ban đầu")
    parser.add_argument("--num-workers", type=int, default=None, help="Số tiến trình load data")
    parser.add_argument("--debug", action="store_true", help="Giới hạn số mẫu để debug")
    parser.add_argument("--show-sample", action="store_true", help="Bật hiển thị mẫu suy luận thử khi lưu checkpoint")
    args = parser.parse_args()

    # Load config từ YAML
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ưu tiên args từ command line, nếu không có thì lấy từ config
    train_dir = args.train_dir or config["paths"]["synthetic_train"]
    val_dir = args.val_dir or config["paths"]["synthetic_val"]
    charset_path = args.charset or config["paths"]["charset"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs_phase1"] + config["training"]["epochs_phase2"]
    lr = args.lr if args.lr is not None else config["training"]["lr"]
    num_workers = args.num_workers if args.num_workers is not None else config["training"]["num_workers"]
    target_h = config["preprocess"]["target_h"]
    best_weight_path = Path(config["paths"]["best_weight"])
    best_weight_path.parent.mkdir(parents=True, exist_ok=True)
    
    lstm_hidden = config["model"]["lstm_hidden"]
    lstm_layers = config["model"]["lstm_layers"]
    lstm_dropout = config["model"]["lstm_dropout"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Khởi chạy huấn luyện trên thiết bị: {device}")
    
    # Tạo thư mục lưu Checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # 1. Khởi tạo DataLoader
    logger.info("Đang load dữ liệu...")
    train_loader, val_loader = get_dataloaders(
        train_dirs=[train_dir],
        val_dirs=[val_dir],
        charset_path=charset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        target_h=target_h
    )

    if args.debug:
        sample = 640
        logger.info(f"DEBUG MODE: Cắt tập dữ liệu xuống còn {sample} mẫu!")
        train_loader.dataset.samples = train_loader.dataset.samples[:sample]
        val_loader.dataset.samples = val_loader.dataset.samples[:sample]
    
    logger.info(f"Số batch train: {len(train_loader)} | Số batch val: {len(val_loader)}")

    # Khởi tạo model dựa trên tham số config
    codec = CharsetCodec(charset_path)
    model = CRNN(
        num_classes=len(codec),
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout
    ).to(device)
    
    # 3. Khởi tạo Criterion, Optimizer, Scheduler
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    # Giảm LR đi một nửa nếu Val Loss không cải thiện sau 5 epochs
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Chuẩn bị file log metric
    log_file_path = Path(config["paths"].get("reports", "reports")) / "training_log_cer.csv"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_cer", "lr", "time"])

    # 4. Vòng lặp huấn luyện
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        batch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            label_len = batch["label_len"].to(device)

            optimizer.zero_grad()
            outputs = model(images)  # (B, T, C)
            
            # Yêu cầu của CTCLoss: input có dạng (T, B, C)
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)

            loss = criterion(log_probs, labels.cpu(), input_lengths, label_len.cpu())
            loss.backward()
            
            # Gradient clipping để chống nổ Gradient (Gradient Explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        val_loss, val_cer = evaluate(model, val_loader, criterion, codec, device)
        scheduler.step(val_loss)
        
        elapsed = time.time() - start_time
        
        logger.info(f"Epoch {epoch:03d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer*100:.2f}% | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {elapsed:.1f}s")
                    
        # Lưu log vào CSV
        with open(log_file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_cer:.4f}", f"{optimizer.param_groups[0]['lr']:.6f}", f"{elapsed:.1f}"])
        
        # Save model nếu Val Loss thấp nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = best_weight_path
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  [+] Đã lưu checkpoint mới tốt nhất tại {ckpt_path}")
            
            # Thử decode ngay 1 batch để theo dõi (nếu được bật)
            if args.show_sample:
                model.eval()
                with torch.no_grad():
                    test_batch = next(iter(val_loader))
                    test_out = model(test_batch["image"].to(device))
                    preds = ctc_greedy_decode(test_out, codec.charset)
                    logger.info(f"  [>] Mẫu suy luận thử: '{preds[0]}'")

if __name__ == "__main__":
    main()
