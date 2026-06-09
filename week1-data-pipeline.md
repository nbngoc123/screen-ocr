# Tuần 1 — Data Pipeline

> Mục tiêu: Có đủ data chất lượng để bắt đầu training. Cuối tuần phải có ít nhất **900k synthetic samples** và **dữ liệu thực tế từ ICDAR đã được tích hợp**.

---

## Tổng quan công việc

| Ngày | Task | Output |
|---|---|---|
| Thứ 2 | Setup môi trường, charset | `charset.txt`, env ready |
| Thứ 3 | Synthetic generator cơ bản | ~100k samples/giờ |
| Thứ 4 | Augmentation pipeline | Augmented dataset |
| Thứ 5 | Tích hợp ICDAR Dataset | ~50k real samples |
| Thứ 6 | Dataset loader + validation | `DataLoader` ready cho train |

---

## Ngày 1 — Setup môi trường & Charset

### 1.1 Tạo virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-directml mss Pillow opencv-python albumentations
pip install uiautomation fastapi uvicorn jiwer wandb pyyaml tqdm
```

### 1.2 Liệt kê toàn bộ font Windows

```python
# scripts/list_fonts.py
import glob
from pathlib import Path

FONT_DIRS = [
    r"C:\Windows\Fonts",
    r"C:\Users\{username}\AppData\Local\Microsoft\Windows\Fonts",
]

def get_all_fonts() -> list[str]:
    fonts = []
    for d in FONT_DIRS:
        fonts.extend(glob.glob(f"{d}\\*.ttf"))
        fonts.extend(glob.glob(f"{d}\\*.otf"))
    # Loại bold/italic variant, giữ regular khi có thể
    seen_families = set()
    filtered = []
    for f in fonts:
        name = Path(f).stem.lower()
        # Bỏ variant rõ ràng nếu đã có regular
        if any(s in name for s in ["bold", "italic", "heavy", "black", "light", "thin"]):
            continue
        filtered.append(f)
    print(f"Found {len(filtered)} regular fonts")
    return filtered

if __name__ == "__main__":
    fonts = get_all_fonts()
    Path("data/font_list.txt").write_text("\n".join(fonts))
```

### 1.3 Định nghĩa charset

```python
# src/charset.py
import string
from pathlib import Path

def build_charset() -> str:
    """
    Charset cho bài toán Việt + Anh trên screen.
    ~235 ký tự, đủ dùng, không quá bloat.
    """
    # ASCII cơ bản
    base = string.ascii_letters + string.digits + string.punctuation + " "

    # Ký tự tiếng Việt (Unicode NFC)
    viet_lower = (
        "àáảãạ"
        "ăắằẳẵặ"
        "âấầẩẫậ"
        "èéẻẽẹ"
        "êếềểễệ"
        "ìíỉĩị"
        "òóỏõọ"
        "ôốồổỗộ"
        "ơớờởỡợ"
        "ùúủũụ"
        "ưứừửữự"
        "ỳýỷỹỵ"
        "đ"
    )
    viet_upper = viet_lower.upper()

    # Ký tự UI phổ biến không nằm trong ASCII printable
    ui_extras = "…•·×÷±°©®™€£¥₫→←↑↓↔"

    full = sorted(set(base + viet_lower + viet_upper + ui_extras))

    # Thêm CTC blank token ở index 0
    charset_str = "".join(full)
    print(f"Charset size: {len(charset_str)} ký tự")
    return charset_str


def save_charset(path: str = "data/charset.txt") -> None:
    charset = build_charset()
    Path(path).write_text(charset, encoding="utf-8")
    print(f"Saved charset to {path}")


def load_charset(path: str = "data/charset.txt") -> str:
    return Path(path).read_text(encoding="utf-8")


class CharsetCodec:
    """Encode/decode string ↔ index list cho CTC."""

    def __init__(self, charset_path: str = "data/charset.txt"):
        self.charset = load_charset(charset_path)
        # index 0 = CTC blank
        self.char2idx = {c: i + 1 for i, c in enumerate(self.charset)}
        self.idx2char = {i + 1: c for i, c in enumerate(self.charset)}
        self.blank_idx = 0

    def encode(self, text: str) -> list[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices: list[int]) -> str:
        # CTC collapse: loại blank và ký tự lặp liên tiếp
        result = []
        prev = self.blank_idx
        for idx in indices:
            if idx != self.blank_idx and idx != prev:
                result.append(self.idx2char.get(idx, ""))
            prev = idx
        return "".join(result)

    def __len__(self) -> int:
        return len(self.charset)
```

---

## Ngày 2 — Synthetic Data Generator

### 2.1 Text corpus cho generation

```python
# src/corpus.py
import random
import string

# Danh sách text mẫu theo domain
DOMAINS = {
    "ui_labels": [
        "OK", "Cancel", "Apply", "Close", "Save", "Open", "Delete",
        "Settings", "Help", "About", "File", "Edit", "View", "Tools",
        "Cài đặt", "Đóng", "Lưu", "Mở file", "Xoá", "Thoát",
    ],
    "sentences_vn": [
        "Xin chào người dùng", "Tổng cộng: 1.234.567 VND",
        "Ngày tạo: 09/06/2024", "Trạng thái: Đang xử lý",
        "Mã đơn hàng: ORD-00421", "Họ tên: Nguyễn Văn A",
        "Email: example@gmail.com", "Điện thoại: 0912 345 678",
    ],
    "sentences_en": [
        "Welcome to the system", "Total: $1,234.56",
        "Status: Processing", "Order ID: ORD-00421",
        "Please enter your password", "File not found",
        "Connection established", "Loading... please wait",
    ],
    "code_snippets": [
        "if __name__ == '__main__':", "import numpy as np",
        "def forward(self, x):", "return torch.sigmoid(x)",
        "print(f'Loss: {loss:.4f}')", "model.train()",
    ],
    "numbers": [
        "1,234,567", "3.14159", "0x1A2F", "100%",
        "08:30:00", "2024-06-09", "v1.2.3", "#FF5733",
    ],
}


def random_text(min_len: int = 2, max_len: int = 40) -> str:
    """Sinh text ngẫu nhiên từ corpus hoặc generate tổng hợp."""
    if random.random() < 0.7:
        # Lấy từ corpus
        domain = random.choice(list(DOMAINS.keys()))
        text = random.choice(DOMAINS[domain])
    else:
        # Generate ngẫu nhiên
        length = random.randint(min_len, max_len)
        chars = string.ascii_letters + string.digits + " "
        text = "".join(random.choices(chars, k=length)).strip()

    # Truncate nếu quá dài
    return text[:max_len] if len(text) > max_len else text
```

### 2.2 Image generator

```python
# src/data_gen.py
import random
import numpy as np
from pathlib import Path
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
from tqdm import tqdm

from src.corpus import random_text


@lru_cache(maxsize=512)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _rand_color(lo: int, hi: int) -> tuple[int, int, int]:
    return tuple(random.randint(lo, hi) for _ in range(3))


def _ensure_contrast(bg: tuple, fg: tuple, min_diff: int = 80) -> tuple:
    """Đảm bảo fg đủ contrast so với bg."""
    diff = sum(abs(b - f) for b, f in zip(bg, fg)) / 3
    if diff < min_diff:
        # Invert fg
        fg = tuple(255 - c for c in fg)
    return fg


def generate_sample(
    text: str,
    font_path: str,
    font_size: int | None = None,
    padding: int = 8,
) -> np.ndarray | None:
    """
    Sinh ảnh chứa text.
    Returns: numpy array BGR hoặc None nếu lỗi.
    """
    size = font_size or random.randint(10, 52)
    font = _load_font(font_path, size)

    try:
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0] + padding * 2
        h = bbox[3] - bbox[1] + padding * 2
        if w < 4 or h < 4:
            return None
    except Exception:
        return None

    # Background
    bg_mode = random.choice(["solid", "light_gradient_sim", "near_white"])
    if bg_mode == "solid":
        bg = _rand_color(180, 255)
    elif bg_mode == "near_white":
        bg = _rand_color(230, 255)
    else:
        bg = _rand_color(200, 245)

    # Foreground
    fg = _rand_color(0, 80) if random.random() < 0.8 else _rand_color(150, 255)
    fg = _ensure_contrast(bg, fg)

    img = Image.new("RGB", (w, h), color=bg)
    ImageDraw.Draw(img).text((padding, padding - bbox[1]), text, font=font, fill=fg)

    # Convert sang numpy BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_dataset(
    font_list: list[str],
    output_dir: str,
    n_samples: int = 500_000,
    font_size_range: tuple[int, int] = (10, 52),
) -> None:
    """Sinh toàn bộ synthetic dataset."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = []
    count = 0

    with tqdm(total=n_samples, desc="Generating") as pbar:
        while count < n_samples:
            font_path = random.choice(font_list)
            text = random_text()
            if not text:
                continue

            img = generate_sample(
                text, font_path,
                font_size=random.randint(*font_size_range),
            )
            if img is None:
                continue

            img_path = out / f"{count:07d}.png"
            cv2.imwrite(str(img_path), img)
            meta.append({"file": f"{count:07d}.png", "label": text})

            count += 1
            pbar.update(1)

    # Lưu metadata vào jsonl
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Generated {count} samples → {output_dir}")
```

### 2.3 Chạy generation

```bash
# scripts/run_gen.py
python - <<'EOF'
from src.data_gen import generate_dataset
from pathlib import Path

fonts = Path("data/font_list.txt").read_text().splitlines()
fonts = [f for f in fonts if f.strip()]

generate_dataset(
    font_list=fonts,
    output_dir="data/synthetic/train",
    n_samples=480_000,
)
generate_dataset(
    font_list=fonts[:int(len(fonts)*0.1)],   # font mới cho val
    output_dir="data/synthetic/val",
    n_samples=60_000,
)
EOF
```

> **Benchmark:** Trên CPU thường sinh được ~80k–120k samples/giờ. Để đạt 500k trong một ngày, chạy song song bằng `multiprocessing`.

```python
# Parallel generation
from multiprocessing import Pool

def gen_chunk(args):
    fonts, out_dir, n, offset = args
    # generate n samples, đánh index từ offset
    ...

chunks = [(fonts, "data/synthetic/train", 100_000, i*100_000) for i in range(5)]
with Pool(5) as p:
    p.map(gen_chunk, chunks)
```

---

## Ngày 3 — Augmentation Pipeline

### 3.1 Augmentation module

```python
# src/augment.py
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pipeline cho training — moderate augmentation
TRAIN_TRANSFORM = A.Compose([
    A.GaussianBlur(blur_limit=(1, 3), p=0.25),
    A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.3,
    ),
    A.GaussNoise(var_limit=(5.0, 25.0), mean=0, p=0.2),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.15),
    # Normalize về [-1, 1]
    A.Normalize(mean=(0.5,), std=(0.5,)),
])

# Pipeline cho val/test — chỉ normalize
VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
])


def preprocess_for_crnn(
    image: np.ndarray,
    target_h: int = 32,
    is_train: bool = True,
) -> np.ndarray:
    """
    Chuẩn bị ảnh cho CRNN:
    - Resize về H=32, giữ aspect ratio
    - Convert grayscale
    - Augment nếu train
    - Normalize [-1, 1]
    Returns: (1, H, W) float32
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image size: {w}×{h}")

    # Resize
    new_w = max(int(w * target_h / h), 1)
    image = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Augment
    transform = TRAIN_TRANSFORM if is_train else VAL_TRANSFORM
    augmented = transform(image=image[..., np.newaxis])["image"]

    return augmented.squeeze(-1)  # (H, W) float32
```

### 3.2 Kiểm tra augmentation visually

```python
# scripts/visualize_aug.py
import cv2
import matplotlib.pyplot as plt
from src.data_gen import generate_sample
from src.augment import preprocess_for_crnn
import random
from pathlib import Path

fonts = Path("data/font_list.txt").read_text().splitlines()

fig, axes = plt.subplots(4, 6, figsize=(18, 10))
for row in axes:
    font = random.choice(fonts)
    text = "Xin chào 123 OCR"
    orig = generate_sample(text, font, font_size=24)
    row[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    row[0].set_title("Original")
    for i in range(1, 6):
        aug = preprocess_for_crnn(orig, is_train=True)
        row[i].imshow(aug, cmap="gray")
        row[i].set_title(f"Aug {i}")
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.savefig("reports/augmentation_check.png")
print("Saved reports/augmentation_check.png")
```

---

## Ngày 4 — Tích hợp Dữ liệu Thực tế (ICDAR)

Thay vì dùng Crawler UIAutomation (khá tốn thời gian chạy nền và dễ nhiễu), chúng ta sử dụng dữ liệu thực tế đã được cắt sẵn từ bộ ICDAR 2015/2017 Robust Reading (đã được parse thành file labels.jsonl chuẩn). Dữ liệu này cung cấp các mẫu văn bản và số thực tế trên màn hình, giúp mô hình mạnh mẽ và thực tế hơn rất nhiều.

---

## Ngày 5 — Dataset Loader & Validation

### 5.1 PyTorch Dataset

```python
# src/dataset.py
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.charset import CharsetCodec
from src.augment import preprocess_for_crnn


class OCRDataset(Dataset):
    def __init__(
        self,
        data_dirs: list[str],
        charset_path: str = "data/charset.txt",
        is_train: bool = True,
        max_label_len: int = 80,
        target_h: int = 32,
    ):
        self.codec = CharsetCodec(charset_path)
        self.is_train = is_train
        self.max_label_len = max_label_len
        self.target_h = target_h
        self.samples = self._load_samples(data_dirs)
        print(f"Dataset: {len(self.samples)} samples, train={is_train}")

    def _load_samples(self, data_dirs: list[str]) -> list[dict]:
        samples = []
        for d in data_dirs:
            p = Path(d)
            labels_file = p / "labels.jsonl"
            if labels_file.exists():
                # Synthetic format
                with open(labels_file, encoding="utf-8") as f:
                    for line in f:
                        m = json.loads(line)
                        img_path = p / m["file"]
                        if img_path.exists():
                            samples.append({"img": str(img_path), "label": m["label"]})
            else:
                # Real format (paired .png + .json)
                for img_path in sorted(p.glob("*.png")):
                    json_path = img_path.with_suffix(".json")
                    if json_path.exists():
                        meta = json.loads(json_path.read_text(encoding="utf-8"))
                        samples.append({"img": str(img_path), "label": meta["label"]})

        random.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img = cv2.imread(sample["img"])
        if img is None:
            # Fallback: blank image
            img = np.ones((32, 100, 3), dtype=np.uint8) * 255

        # Preprocess
        img_tensor = preprocess_for_crnn(img, self.target_h, self.is_train)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()  # (1, H, W)

        # Encode label
        label = sample["label"][:self.max_label_len]
        encoded = self.codec.encode(label)

        return {
            "image":        img_tensor,
            "label":        torch.tensor(encoded, dtype=torch.long),
            "label_len":    torch.tensor(len(encoded), dtype=torch.long),
            "label_str":    label,
            "img_path":     sample["img"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad images về cùng width, pad labels về cùng length."""
    # Pad images (chiều rộng khác nhau)
    max_w = max(b["image"].shape[2] for b in batch)
    images = torch.zeros(len(batch), 1, 32, max_w)
    for i, b in enumerate(batch):
        w = b["image"].shape[2]
        images[i, :, :, :w] = b["image"]

    # Pad labels
    max_l = max(len(b["label"]) for b in batch)
    labels = torch.zeros(len(batch), max_l, dtype=torch.long)
    for i, b in enumerate(batch):
        l = len(b["label"])
        labels[i, :l] = b["label"]

    return {
        "image":       images,
        "label":       labels,
        "label_len":   torch.stack([b["label_len"] for b in batch]),
        "label_str":   [b["label_str"] for b in batch],
        "input_len":   torch.tensor(
            [max_w // 4 for _ in batch], dtype=torch.long  # sau CNN stride 4
        ),
    }


def get_dataloaders(
    train_dirs: list[str],
    val_dirs: list[str],
    charset_path: str = "data/charset.txt",
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:

    train_ds = OCRDataset(train_dirs, charset_path, is_train=True)
    val_ds   = OCRDataset(val_dirs,   charset_path, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
```

### 5.2 Validation dataset

```python
# scripts/validate_dataset.py
"""Kiểm tra dataset trước khi train — phát hiện lỗi sớm."""
import cv2
import json
import numpy as np
from pathlib import Path
from collections import Counter
from src.charset import CharsetCodec

codec = CharsetCodec("data/charset.txt")


def validate_split(data_dir: str, split_name: str, max_check: int = 10_000):
    p = Path(data_dir)
    labels_file = p / "labels.jsonl"
    samples = []

    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"\n=== {split_name} ({len(samples)} samples) ===")

    errors = []
    label_lens = []
    oov_chars = Counter()
    img_sizes = []

    for m in samples[:max_check]:
        img_path = p / m["file"]
        label = m["label"]

        # Kiểm tra file tồn tại
        if not img_path.exists():
            errors.append(f"Missing: {img_path}")
            continue

        # Kiểm tra ảnh readable
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"Corrupt: {img_path}")
            continue

        h, w = img.shape[:2]
        if h < 4 or w < 4:
            errors.append(f"Too small {w}×{h}: {img_path}")
            continue

        img_sizes.append((w, h))
        label_lens.append(len(label))

        # Kiểm tra OOV chars
        for c in label:
            if c not in codec.charset:
                oov_chars[c] += 1

    # Report
    print(f"  Errors: {len(errors)}")
    for e in errors[:5]:
        print(f"    {e}")

    if img_sizes:
        ws = [s[0] for s in img_sizes]
        hs = [s[1] for s in img_sizes]
        print(f"  Image W: min={min(ws)}, max={max(ws)}, mean={sum(ws)/len(ws):.0f}")
        print(f"  Image H: min={min(hs)}, max={max(hs)}, mean={sum(hs)/len(hs):.0f}")

    if label_lens:
        print(f"  Label len: min={min(label_lens)}, max={max(label_lens)}, "
              f"mean={sum(label_lens)/len(label_lens):.1f}")

    if oov_chars:
        print(f"  OOV chars ({len(oov_chars)} types): {dict(oov_chars.most_common(10))}")
        print("  → Cân nhắc thêm vào charset hoặc filter những sample này")
    else:
        print("  OOV chars: none (OK)")

    return len(errors) == 0


if __name__ == "__main__":
    ok = True
    ok &= validate_split("data/synthetic/train", "Train")
    ok &= validate_split("data/synthetic/val",   "Val")
    ok &= validate_split("data/raw/Recognition", "ICDAR")
    print("\n" + ("Dataset OK" if ok else "Dataset có lỗi — sửa trước khi train"))
```

---

## Checklist cuối tuần 1

- [ ] `charset.txt` đã tạo, có đủ ký tự Việt + UI extras
- [ ] 800k train samples đã sinh xong (`data/synthetic/train/`)
- [ ] 100k val samples đã sinh xong (`data/synthetic/val/`)
- [ ] Val set dùng font khác train set (verify bằng `set` intersection)
- [ ] Tích hợp thành công dữ liệu thực tế (ICDAR Recognition)
- [ ] `validate_dataset.py` chạy không ra error
- [ ] OOV chars < 1% trên toàn dataset
- [ ] `DataLoader` test: load 1 batch không báo lỗi
- [ ] Augmentation visualized và trông hợp lý

---

## Metrics cuối tuần 1

| Metric | Target | Cách đo |
|---|---|---|
| Tổng synthetic samples | ≥ 900k | `wc -l data/synthetic/train/labels.jsonl` |
| Real samples (ICDAR) | ≥ 20k | `wc -l data/raw/Recognition/labels.jsonl` |
| OOV rate | < 1% | `validate_dataset.py` |
| DataLoader throughput | ≥ 500 samples/s | `time_dataloader.py` |
| Corrupt images | 0 | `validate_dataset.py` |
