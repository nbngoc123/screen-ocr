"""
model.py — CRNN architecture cho Text Recognition.

Kiến trúc: ResNet18 (CNN backbone) → BiLSTM → FC → CTC decode

    Image (1×32×W)
      → ResNet18 truncated, stride 1 ở layer4  → feature map (512×1×W')
      → Squeeze + Permute                       → sequence (W'×512)
      → BiLSTM (2 layers, hidden=256)           → (W'×512)
      → Linear                                  → (W'×num_classes+1)
      → CTC decode                              → string

Quy tắc (screen-ocr-rules.md §3.4):
    CNN backbone khởi tạo từ pretrained ImageNet.
    Chỉ random init BiLSTM và FC head.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CRNN(nn.Module):
    """
    CRNN model cho text recognition.

    Sử dụng ResNet18 pretrained làm CNN backbone, không dùng
    avgpool và fc gốc. Stride ở layer4 được đổi thành 1 để
    giữ độ phân giải theo chiều ngang.

    Args:
        num_classes: Số ký tự trong charset (không tính CTC blank).
                     Output FC sẽ có num_classes + 1 units.

    Example:
        model = CRNN(num_classes=235)
        x = torch.randn(4, 1, 32, 200)  # batch=4, gray, H=32, W=200
        out = model(x)  # (4, W', 236)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # TODO: implement — xem screen-ocr-project.md §CRNN architecture
        # 1. Load resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 2. Sửa stride layer4: conv1.stride = (1,1), downsample[0].stride = (1,1)
        # 3. Xây dựng self.cnn = Sequential(backbone layers bỏ avgpool + fc)
        # 4. self.rnn = LSTM(input=512, hidden=256, layers=2, bidirectional=True)
        # 5. self.fc  = Linear(512, num_classes + 1)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, 32, W) — grayscale, H=32.

        Returns:
            Logits tensor (B, W', num_classes+1) chưa softmax.
        """
        # TODO: implement
        raise NotImplementedError


def ctc_greedy_decode(logits: torch.Tensor, charset: str) -> list[str]:
    """
    Greedy CTC decode — argmax theo time dimension rồi collapse.

    Args:
        logits:  (B, T, C) tensor, chưa softmax.
        charset: Chuỗi charset (không tính blank ở index 0).

    Returns:
        List[str] — text đã decode cho mỗi item trong batch.
    """
    # TODO: implement
    raise NotImplementedError
