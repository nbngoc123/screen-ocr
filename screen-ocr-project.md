# Screen OCR Engine — Tài liệu dự án

> Hệ thống OCR tự build cho màn hình Windows, phục vụ đa bài toán: automation, RAG, translation, game modding.

---

## Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Loại dự án | Custom OCR Engine (train từ đầu) |
| Nền tảng | Windows 10/11 |
| Ngôn ngữ | Python 3.11+ |
| Framework ML | PyTorch + ONNX Runtime |
| Kiến trúc | 2-stage: DBNet++ (detect) + CRNN (recognize) |
| Mục tiêu CER | < 2% trên screen text |
| Latency mục tiêu | < 100ms/frame end-to-end |

---

## Mục tiêu

Xây dựng một OCR engine chạy offline trên Windows, nhận đầu vào là ảnh chụp màn hình (`mss`), trả ra text có cấu trúc để feed vào các downstream task:

- **RAG agent** — index nội dung màn hình vào vector DB (ChromaDB/FAISS)
- **UI Automation** — tìm và tương tác với element theo text
- **Translation pipeline** — đọc rồi dịch nội dung màn hình
- **Game modding** — đọc UI text game realtime

---

## Kiến trúc hệ thống

### Tổng quan 2-stage pipeline

```
[mss capture] → [DBNet++ detect] → [crop ROIs] → [CRNN recognize] → [post-process] → [text output]
```

### Stage 1 — Text Detection (DBNet++)

Phát hiện bounding box của từng vùng text trên ảnh màn hình.

- **Model:** DBNet++ (pretrained ICDAR, fine-tune trên screen data)
- **Input:** Screenshot RGB, bất kỳ resolution
- **Output:** Danh sách polygon/bbox của các vùng text
- **Tốc độ:** ~15ms/frame trên GPU

```python
# Inference detection
from dbnet import DBNetDetector

detector = DBNetDetector("dbnet_screen.onnx")
boxes = detector.detect(screenshot_np)  # list of [(x1,y1,x2,y2), ...]
```

### Stage 2 — Text Recognition (CRNN)

Đọc text từ từng ROI đã crop.

- **Model:** CRNN (ResNet18 backbone + BiLSTM + CTC decoder)
- **Input:** Ảnh crop, resize về H=32px, width giữ tỉ lệ
- **Output:** String text + confidence score
- **Tốc độ:** ~5ms/crop trên GPU, ~15ms trên CPU

**Kiến trúc chi tiết:**

```
Image (3×32×W)
  → ResNet18 (truncated, stride 1 tầng cuối)  → feature map (512×1×W')
  → Squeeze + Permute                          → sequence (W'×512)
  → BiLSTM (2 layers, hidden=256)             → (W'×512)
  → Linear                                     → (W'×num_classes+1)
  → CTC decode                                 → string
```

### Full inference flow

```python
import mss
import numpy as np
from detector import DBNetDetector
from recognizer import CRNNRecognizer
from postprocess import postprocess

detector   = DBNetDetector("models/dbnet.onnx")
recognizer = CRNNRecognizer("models/crnn.onnx", charset_path="charset.txt")

with mss.mss() as sct:
    mon = sct.monitors[1]  # primary monitor
    frame = np.array(sct.grab(mon))

boxes = detector.detect(frame)

results = []
for box in boxes:
    crop  = frame[box.y1:box.y2, box.x1:box.x2]
    text, conf = recognizer.recognize(crop)
    text  = postprocess(text)
    if conf > 0.6:
        results.append({"box": box, "text": text, "conf": conf})
```

---

## Data Pipeline

### 1. Synthetic data generation

Nguồn data chính — có thể sinh vô hạn, ground truth 100% chính xác.

**Quy trình:**

1. Load toàn bộ font từ `C:\Windows\Fonts` (`.ttf`, `.otf`)
2. Sinh text ngẫu nhiên: Việt, Anh, số, ký hiệu, mã hoá đơn, v.v.
3. Render lên canvas với background/foreground ngẫu nhiên
4. Lưu cặp `(image, label)`

```python
from PIL import Image, ImageDraw, ImageFont
import random, glob, os

FONTS = glob.glob(r"C:\Windows\Fonts\*.ttf")

def gen_sample(text: str, font_size: int = None) -> tuple[Image.Image, str]:
    font_path = random.choice(FONTS)
    size      = font_size or random.randint(10, 48)
    font      = ImageFont.truetype(font_path, size)

    bbox = font.getbbox(text)
    w, h = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 10

    bg  = tuple(random.randint(200, 255) for _ in range(3))
    fg  = tuple(random.randint(0, 80) for _ in range(3))
    img = Image.new("RGB", (w, h), color=bg)
    ImageDraw.Draw(img).text((10, 5), text, font=font, fill=fg)

    return img, text
```

**Augmentation (albumentations):**

```python
import albumentations as A

aug = A.Compose([
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.15, p=0.3),
    A.GaussNoise(var_limit=(5, 20), p=0.2),
    # KHÔNG dùng rotation/perspective — screen text luôn thẳng
])
```

### 2. Real screen capture với UIAutomation

Thu thập real data tự động — không cần label tay.

```python
import uiautomation as auto
import mss
import numpy as np
from pathlib import Path

def collect_screen_samples(output_dir: str, max_samples: int = 10000):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    count = 0

    with mss.mss() as sct:
        screenshot = np.array(sct.grab(sct.monitors[1]))

    def walk(ctrl, depth=0):
        nonlocal count
        if count >= max_samples:
            return
        try:
            name = ctrl.Name.strip()
            rect = ctrl.BoundingRectangle
            if name and len(name) >= 2 and rect.width() > 5 and rect.height() > 5:
                crop = screenshot[rect.top:rect.bottom, rect.left:rect.right]
                if crop.size > 0:
                    img_path = f"{output_dir}/{count:06d}.png"
                    lbl_path = f"{output_dir}/{count:06d}.txt"
                    cv2.imwrite(img_path, crop)
                    Path(lbl_path).write_text(name, encoding="utf-8")
                    count += 1
        except Exception:
            pass
        for child in ctrl.GetChildren():
            walk(child, depth + 1)

    walk(auto.GetRootControl())
    print(f"Collected {count} samples")
```

### 3. Charset definition

Charset cho bài toán Việt + Anh trên screen (~230 ký tự):

```python
import string

BASE     = string.ascii_letters + string.digits + string.punctuation + " "
VIET     = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
VIET    += VIET.upper()
CHARSET  = sorted(set(BASE + VIET))

# Save
with open("charset.txt", "w", encoding="utf-8") as f:
    f.write("".join(CHARSET))
```

### 4. Dataset split

| Split | Tỉ lệ | Số lượng | Ghi chú |
|---|---|---|---|
| Train | 80% | ~480k | Synthetic chủ yếu |
| Val   | 10% | ~60k  | Có font chưa thấy trong train |
| Test  | 10% | ~60k  | Real screen capture |

---

## Training

### CRNN — Kiến trúc đầy đủ

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18

class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet18(pretrained=True)
        # Bỏ layer4 avgpool và fc, giữ stride 1 ở layer4
        backbone.layer4[0].conv1.stride  = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.cnn = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.1,
        )
        self.fc = nn.Linear(512, num_classes + 1)  # +1 CTC blank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)                  # (B, 512, 1, W')
        feat = feat.squeeze(2)              # (B, 512, W')
        feat = feat.permute(0, 2, 1)        # (B, W', 512)
        out, _ = self.rnn(feat)             # (B, W', 512)
        return self.fc(out)                 # (B, W', num_classes+1)
```

### Training loop

```python
from torch.cuda.amp import GradScaler, autocast

model     = CRNN(num_classes=len(CHARSET)).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=50, steps_per_epoch=len(train_loader)
)
ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)
scaler    = GradScaler()

for epoch in range(50):
    for images, labels, label_lengths in train_loader:
        images = images.cuda()

        with autocast():
            logits = model(images)                      # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
```

### Hyperparameters

| Parameter | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 (OneCycleLR) |
| Batch size | 256 (AMP) |
| Image height | 32px (cố định) |
| Image width | dynamic |
| Epochs | 50 synthetic + 10 fine-tune |
| Weight decay | 1e-4 |
| Grad clip | 5.0 |

### Curriculum learning

| Giai đoạn | Epoch | Data | Mô tả |
|---|---|---|---|
| 1 — Foundation | 1–15 | Synthetic, font lớn, bg trắng | Học charset cơ bản |
| 2 — Robustness | 16–35 | Synthetic + augmentation mạnh | Font nhỏ, bg phức tạp |
| 3 — Fine-tune | 36–45 | Real screen capture | lr = 1e-4 |

### Metrics

```python
from jiwer import cer, wer

def evaluate(model, dataloader):
    all_pred, all_gt = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images.cuda())
            preds  = ctc_greedy_decode(logits, CHARSET)
            all_pred.extend(preds)
            all_gt.extend(labels)

    return {
        "CER":         cer(all_gt, all_pred),
        "WER":         wer(all_gt, all_pred),
        "ExactMatch":  sum(p == g for p, g in zip(all_pred, all_gt)) / len(all_gt),
    }
```

**Target:** CER < 2%, ExactMatch > 90% trên val set screen text.

---

## Post-processing

```python
import re

def postprocess(text: str) -> str:
    # 1. Whitespace normalize
    text = re.sub(r'\s+', ' ', text).strip()
    # 2. OCR common errors
    text = re.sub(r'\b([A-Z]{2,})0([A-Z0-9])', r'\g<1>O\g<2>', text)
    text = re.sub(r'(?<=[0-9])O(?=[0-9])', '0', text)
    # 3. Domain-specific (mở rộng tùy bài toán)
    text = re.sub(
        r'([\w.+\-]+)\s*@\s*([\w.\-]+\s*\.\s*[a-z]{2,})',
        lambda m: m.group(1) + '@' + m.group(2).replace(' ', ''),
        text
    )
    return text
```

---

## Deployment

### Export ONNX

```python
dummy = torch.randn(1, 3, 32, 200).cuda()

torch.onnx.export(
    model, dummy, "models/crnn.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {3: "width"}, "logits": {1: "time"}},
    opset_version=17,
    do_constant_folding=True,
)
```

### ONNX Runtime inference (DirectML)

```python
import onnxruntime as ort
import numpy as np
import cv2

sess = ort.InferenceSession(
    "models/crnn.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
)

def recognize(img_bgr: np.ndarray) -> tuple[str, float]:
    # Preprocess
    h, w = img_bgr.shape[:2]
    new_w = max(int(w * 32 / h), 1)
    img   = cv2.resize(img_bgr, (new_w, 32))
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img   = img.astype(np.float32) / 255.0
    img   = (img - 0.5) / 0.5
    x     = img[np.newaxis, np.newaxis, :, :]  # (1,1,32,W)

    logits = sess.run(None, {"image": x})[0]    # (1,T,C)
    text, conf = ctc_greedy_decode(logits[0])
    return text, conf
```

### FastAPI service

```python
from fastapi import FastAPI
from pydantic import BaseModel
import base64, numpy as np, cv2

app = FastAPI()

class OCRRequest(BaseModel):
    image_b64: str
    conf_threshold: float = 0.6

@app.post("/recognize")
def recognize_endpoint(req: OCRRequest):
    img_bytes = base64.b64decode(req.image_b64)
    img_np    = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    text, conf = recognize(img_np)
    if conf < req.conf_threshold:
        return {"text": "", "conf": conf, "filtered": True}
    return {"text": postprocess(text), "conf": conf, "filtered": False}
```

---

## Cấu trúc thư mục

```
screen-ocr/
├── data/
│   ├── synthetic/          # Generated samples
│   │   ├── 000001.png
│   │   └── 000001.txt
│   ├── real/               # UIAutomation captures
│   └── charset.txt
├── models/
│   ├── dbnet.onnx
│   ├── crnn.onnx
│   └── crnn_checkpoint.pt
├── src/
│   ├── data_gen.py         # Synthetic generation
│   ├── crawler.py          # UIAutomation collector
│   ├── dataset.py          # PyTorch Dataset
│   ├── model.py            # CRNN architecture
│   ├── train.py            # Training loop
│   ├── detector.py         # DBNet wrapper
│   ├── recognizer.py       # CRNN ONNX wrapper
│   ├── postprocess.py      # Text cleanup
│   └── pipeline.py         # Full end-to-end
├── api/
│   └── server.py           # FastAPI service
├── scripts/
│   ├── export_onnx.py
│   └── evaluate.py
├── configs/
│   └── train_config.yaml
└── requirements.txt
```

---

## Requirements

```txt
# requirements.txt
torch>=2.2.0
torchvision>=0.17.0
onnxruntime-directml>=1.17.0
mss>=9.0.0
Pillow>=10.0.0
opencv-python>=4.9.0
albumentations>=1.3.0
uiautomation>=2.0.18
fastapi>=0.110.0
uvicorn>=0.29.0
jiwer>=3.0.0
wandb>=0.16.0
pyyaml>=6.0.0
```

---

## Roadmap

| Tuần | Milestone |
|---|---|
| 1 | Data pipeline: synthetic gen + UIAutomation crawler |
| 2 | Train CRNN baseline, CER < 10% |
| 3 | Fine-tune với real data, CER < 2% |
| 4 | Export ONNX, FastAPI, end-to-end < 100ms |
| 5+ | Continuous improvement, tích hợp downstream |
