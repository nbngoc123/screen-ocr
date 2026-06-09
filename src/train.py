"""
train.py — Training loop cho CRNN model.

Áp dụng curriculum learning 3 giai đoạn:
    Phase 1 (Foundation, epoch 1–15):   Synthetic, font lớn, bg trắng
    Phase 2 (Robustness, epoch 16–35):  Synthetic + augmentation mạnh
    Phase 3 (Fine-tune, epoch 36–45):   Real screen capture, lr thấp

Quy tắc bắt buộc (screen-ocr-rules.md):
    §3.1 — AMP (torch.cuda.amp)
    §3.2 — Gradient clipping (max_norm=5.0)
    §3.3 — Checkpoint theo val CER, không theo epoch
    §3.5 — Sanity check 1 batch trước khi train
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.model import CRNN, ctc_greedy_decode

logger = logging.getLogger(__name__)


def compute_loss(
    model: CRNN,
    batch: dict,
    ctc_loss: nn.CTCLoss,
) -> torch.Tensor:
    """
    Tính CTC loss cho một batch.

    Args:
        model:    CRNN model (train mode).
        batch:    Dict từ collate_fn: image, label, label_len, input_len.
        ctc_loss: nn.CTCLoss instance.

    Returns:
        Scalar loss tensor.
    """
    # TODO: implement
    raise NotImplementedError


def sanity_check(
    model: CRNN,
    dataloader: torch.utils.data.DataLoader,
    ctc_loss: nn.CTCLoss,
) -> None:
    """
    Chạy 1 batch trước khi train để phát hiện bug sớm.

    Kiểm tra:
        - Loss không phải NaN
        - Loss < 20 (nếu quá cao → label encoding sai)

    Args:
        model:      CRNN model.
        dataloader: Training dataloader.
        ctc_loss:   CTCLoss instance.

    Raises:
        RuntimeError: Nếu sanity check không pass.
    """
    # TODO: implement — xem screen-ocr-rules.md §3.5
    raise NotImplementedError


def evaluate(
    model: CRNN,
    dataloader: torch.utils.data.DataLoader,
    charset: str,
    device: torch.device,
) -> dict[str, float]:
    """
    Đánh giá model trên val/test set.

    Args:
        model:      CRNN model (eval mode).
        dataloader: Val/test dataloader.
        charset:    Charset string.
        device:     CPU hoặc CUDA device.

    Returns:
        Dict với keys: "CER", "WER", "ExactMatch"
    """
    # TODO: implement — dùng jiwer.cer + jiwer.wer
    raise NotImplementedError


def maybe_save_checkpoint(
    model: CRNN,
    val_cer: float,
    path: str,
    current_epoch: int,
    best_cer: float,
) -> float:
    """
    Lưu checkpoint nếu val CER tốt hơn best hiện tại.

    Args:
        model:         Model cần save.
        val_cer:       CER hiện tại trên val set.
        path:          Đường dẫn file checkpoint.
        current_epoch: Epoch hiện tại (để log).
        best_cer:      CER tốt nhất trước đó.

    Returns:
        best_cer mới (được cập nhật nếu save).
    """
    # TODO: implement — xem screen-ocr-rules.md §3.3
    raise NotImplementedError


def train(config: dict) -> None:
    """
    Main training loop với curriculum learning.

    Args:
        config: Dict config từ configs/default.yaml (đã load bằng yaml.safe_load).

    Flow:
        1. Khởi tạo model, optimizer, scheduler, loss, scaler
        2. Sanity check
        3. Phase 1 loop (Foundation)
        4. Phase 2 loop (Robustness)
        5. Phase 3 loop (Fine-tune với real data, lr thấp hơn)
        6. Log metrics lên wandb sau mỗi epoch
    """
    # TODO: implement — xem screen-ocr-project.md §Training loop
    raise NotImplementedError
