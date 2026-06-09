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
from torchvision.models import resnet18, ResNet18_Weights

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

    def __init__(self, num_classes: int, lstm_hidden: int = 256, lstm_layers: int = 2, lstm_dropout: float = 0.1) -> None:
        super().__init__()
        # Load resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Sửa stride các layer cuối để giữ nguyên độ phân giải ngang (width) 
        # nhưng downsample chiều cao (height) từ 32 xuống 1.
        # ResNet18 mặc định: conv1(2x2) -> maxpool(2x2) -> layer2(2x2) -> layer3(2x2) -> layer4(2x2)
        # Nếu chiều cao là 32, qua conv1+maxpool còn 8. 
        # Ta cần 3 bước downsample (8 -> 4 -> 2 -> 1) bằng stride (2, 1).
        backbone.layer2[0].conv1.stride = (2, 1)
        backbone.layer2[0].downsample[0].stride = (2, 1)
        
        backbone.layer3[0].conv1.stride = (2, 1)
        backbone.layer3[0].downsample[0].stride = (2, 1)
        
        backbone.layer4[0].conv1.stride = (2, 1)
        if backbone.layer4[0].downsample is not None:
            backbone.layer4[0].downsample[0].stride = (2, 1)

        # Xây dựng self.cnn = Sequential(backbone layers bỏ avgpool + fc)
        self.cnn = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # self.rnn = LSTM(input=512, hidden=256, layers=2, bidirectional=True)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )

        # BiLSTM xuất ra lstm_hidden * 2, map về num_classes + 1 (1 cho blank)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, 32, W) — grayscale, H=32.

        Returns:
            Logits tensor (B, W', num_classes+1) chưa softmax.
        """
        # CNN forward: (B, 1, 32, W) -> (B, 512, 1, W')
        # Tuy nhiên đầu vào ta là (B, 1, 32, W) grayscale, ResNet cần 3 channels.
        # Nên ta lặp lại 3 kênh.
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        feat = self.cnn(x)             # (B, 512, H', W')
        # Ép chiều cao về đúng 1 pixel để hỗ trợ mọi target_h (vd: 32, 48, 64)
        # Thay vì adaptive_avg_pool2d(feat, (1, None)) gây lỗi ONNX, dùng .mean(dim=2, keepdim=True)
        feat = feat.mean(dim=2, keepdim=True)
        feat = feat.squeeze(2)         # (B, 512, W')
        feat = feat.permute(0, 2, 1)   # (B, W', 512)
        
        # RNN forward
        out, _ = self.rnn(feat)        # (B, W', 512)
        
        # Classifier output
        return self.fc(out)            # (B, W', num_classes+1)


def ctc_greedy_decode(logits: torch.Tensor, charset: str) -> list[str]:
    """
    Greedy CTC decode — argmax theo time dimension rồi collapse.

    Args:
        logits:  (B, T, C) tensor, chưa softmax.
        charset: Chuỗi charset (không tính blank ở index 0).

    Returns:
        List[str] — text đã decode cho mỗi item trong batch.
    """
    # 1. Greedy argmax theo phân phối logit
    # logits shape: (B, T, C)
    _, preds = logits.max(2) # (B, T)
    preds = preds.cpu().numpy()
    
    results = []
    for pred in preds:
        char_list = []
        for i, char_idx in enumerate(pred):
            # Nếu ko phải blank (idx=0) và không bị lặp chữ (khác ký tự trước)
            if char_idx != 0 and (not (i > 0 and char_idx == pred[i - 1])):
                # Kiểm tra tránh out of range nếu dùng nhầm model/charset
                if char_idx - 1 < len(charset):
                    char_list.append(charset[char_idx - 1])
                else:
                    char_list.append('?') # Ký tự không xác định
        results.append("".join(char_list))
        
    return results
