"""
plot_metrics.py — Vẽ biểu đồ quá trình huấn luyện từ file log CSV.

Hỗ trợ đọc cả 2 định dạng file log:
- training_log.csv (Chỉ có Loss và LR)
- training_log_cer.csv (Có thêm CER)
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(csv_path: str, output_path: str):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Lỗi: Không tìm thấy file log tại {csv_file}")
        return

    # Đọc dữ liệu
    df = pd.read_csv(csv_file)
    epochs = df["epoch"].values
    
    # Xác định loại biểu đồ (Có CER hay không)
    has_cer = "val_cer" in df.columns
    
    # Thiết lập số lượng biểu đồ
    num_plots = 3 if has_cer else 2
    
    # Vẽ
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    if num_plots == 2:
        # Nếu chỉ có 2 plot, ép kiểu axes thành array để dễ truy cập
        axes = [axes[0], axes[1]]

    # 1. Biểu đồ Loss
    axes[0].plot(epochs, df["train_loss"], label="Train Loss", color="blue", marker="o", markersize=4)
    axes[0].plot(epochs, df["val_loss"], label="Val Loss", color="orange", marker="x", markersize=4)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. Biểu đồ CER (Nếu có)
    if has_cer:
        axes[1].plot(epochs, df["val_cer"] * 100, label="Val CER (%)", color="red", marker="s", markersize=4)
        axes[1].set_title("Validation Character Error Rate (CER)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("CER (%)")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()
        lr_ax = axes[2]
    else:
        lr_ax = axes[1]

    # 3. Biểu đồ Learning Rate
    lr_ax.plot(epochs, df["lr"], label="Learning Rate", color="green", drawstyle="steps-post")
    lr_ax.set_title("Learning Rate Decay")
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("Learning Rate")
    lr_ax.set_yscale("log")
    lr_ax.grid(True, linestyle="--", alpha=0.6)
    lr_ax.legend()

    plt.tight_layout()
    
    # Lưu ra file
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    print(f"✅ Đã lưu biểu đồ thành công tại: {out_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Training Metrics")
    parser.add_argument("--log", type=str, default="reports/training_log_cer.csv", help="Đường dẫn file CSV log")
    parser.add_argument("--out", type=str, default="reports/training_metrics.png", help="Đường dẫn ảnh PNG xuất ra")
    args = parser.parse_args()
    
    plot_metrics(args.log, args.out)
