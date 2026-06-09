# Screen OCR Engine — Quy tắc xây dựng hệ thống

> Tài liệu này định nghĩa các nguyên tắc bắt buộc, best practices, và anti-patterns khi phát triển hệ thống Screen OCR. Mọi thành viên phải đọc trước khi contribute.

---

## 1. Nguyên tắc kiến trúc

### 1.1 Tách biệt Detection và Recognition

**Bắt buộc:** Detection và Recognition phải là hai module độc lập, giao tiếp qua interface rõ ràng.

```python
# ĐÚNG — interface rõ ràng
class TextDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[BoundingBox]: ...

class TextRecognizer(Protocol):
    def recognize(self, crop: np.ndarray) -> tuple[str, float]: ...

# SAI — gộp vào một class
class OCREngine:
    def run(self, image):
        # detect và recognize lẫn lộn → không thể thay thế từng phần
        ...
```

**Lý do:** Khi cần nâng cấp recognition model, không cần đụng đến detection, và ngược lại.

### 1.2 Model chỉ được load một lần

```python
# ĐÚNG — singleton pattern
class CRNNRecognizer:
    _instance = None

    @classmethod
    def get_instance(cls, model_path: str) -> "CRNNRecognizer":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

# SAI — load model mỗi lần gọi
def recognize(image, model_path="crnn.onnx"):
    sess = ort.InferenceSession(model_path)  # load lại mỗi request → chậm
    ...
```

### 1.3 Pipeline phải có timeout

Mọi inference call phải có timeout để tránh blocking downstream.

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds: float):
    def handler(signum, frame):
        raise TimeoutError(f"OCR timeout after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

# Dùng
with timeout(0.5):  # max 500ms per frame
    result = recognizer.recognize(crop)
```

---

## 2. Quy tắc data

### 2.1 Ground truth phải tuyệt đối chính xác

Không chấp nhận label mơ hồ hoặc gần đúng.

| Nguồn data | Độ tin cậy | Cách xác minh |
|---|---|---|
| Synthetic (Pillow render) | 100% | Chính xác theo definition |
| UIAutomation API | ~99% | Verify bằng visual inspection 1% random |
| Manual label | ~95% | Double-label + disagreement check |
| OCR của engine khác làm GT | **KHÔNG DÙNG** | Circular dependency |

### 2.2 Không dùng output của OCR engine khác làm ground truth

```python
# SAI — dùng Tesseract để tạo label cho training data
for img in unlabeled_images:
    label = pytesseract.image_to_string(img)  # KHÔNG làm thế này
    save_label(img, label)
```

Lý do: model sẽ học cả lỗi của Tesseract, không bao giờ vượt qua được baseline.

### 2.3 Val/Test set phải chứa font chưa thấy trong train

```python
ALL_FONTS = get_all_windows_fonts()
random.shuffle(ALL_FONTS)

TRAIN_FONTS = ALL_FONTS[:int(len(ALL_FONTS) * 0.8)]
VAL_FONTS   = ALL_FONTS[int(len(ALL_FONTS) * 0.8):int(len(ALL_FONTS) * 0.9)]
TEST_FONTS  = ALL_FONTS[int(len(ALL_FONTS) * 0.9):]

# Strict separation — không overlap bất kỳ font nào
assert not set(TRAIN_FONTS) & set(VAL_FONTS)
assert not set(TRAIN_FONTS) & set(TEST_FONTS)
```

### 2.4 Không augment perspective/rotation mạnh trên screen data

Screen text luôn nằm ngang, không bị perspective warp.

```python
# ĐÚNG — augmentation phù hợp screen text
ALLOWED_AUGMENTS = [
    "gaussian_blur",        # subpixel smoothing
    "jpeg_compression",     # screenshot artifact
    "brightness_contrast",  # monitor calibration khác nhau
    "gaussian_noise",       # camera capture
    "slight_shear",         # max ±3 degree
]

# SAI — augmentation không phù hợp
FORBIDDEN_AUGMENTS = [
    "rotation > 5 degree",    # screen text không bị xoay nhiều
    "perspective_transform",  # không xảy ra với screenshot
    "elastic_transform",      # distort chữ phi thực tế
]
```

### 2.5 Logging mọi sample bị lỗi

```python
import logging
from pathlib import Path

error_logger = logging.getLogger("ocr.errors")

def save_error_sample(image: np.ndarray, pred: str, gt: str, conf: float):
    """Log sample bị nhận sai để review và add vào training data."""
    if pred != gt:
        error_dir = Path("data/errors")
        error_dir.mkdir(exist_ok=True)
        idx = len(list(error_dir.glob("*.png")))
        cv2.imwrite(str(error_dir / f"{idx:06d}.png"), image)
        (error_dir / f"{idx:06d}.json").write_text(
            json.dumps({"pred": pred, "gt": gt, "conf": conf}),
            encoding="utf-8"
        )
```

---

## 3. Quy tắc model & training

### 3.1 Luôn dùng AMP (Automatic Mixed Precision)

```python
# ĐÚNG
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

with autocast():
    loss = compute_loss(model, batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# SAI — bỏ AMP → chậm gấp 2, tốn VRAM gấp đôi
loss = compute_loss(model, batch)
loss.backward()
optimizer.step()
```

### 3.2 Gradient clipping bắt buộc

CTC loss có thể explode gradient, đặc biệt đầu training.

```python
# Luôn clip trước khi step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

### 3.3 Checkpoint theo val CER, không theo epoch

```python
best_cer = float("inf")

def maybe_save_checkpoint(model, val_cer: float, path: str):
    global best_cer
    if val_cer < best_cer:
        best_cer = val_cer
        torch.save({
            "model_state": model.state_dict(),
            "val_cer": val_cer,
            "epoch": current_epoch,
        }, path)
        print(f"Saved checkpoint — CER: {val_cer:.4f}")
```

### 3.4 Không train lại từ đầu khi có pretrained

Luôn khởi tạo CNN backbone từ pretrained ImageNet, chỉ random init BiLSTM và FC.

```python
from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # ĐÚNG

backbone = resnet18(weights=None)  # SAI — mất đi feature extraction tốt
```

### 3.5 Validate trước khi train (sanity check)

```python
def sanity_check(model, dataloader, charset):
    """Chạy 1 batch trước khi train để phát hiện bug sớm."""
    model.train()
    batch = next(iter(dataloader))
    try:
        loss = compute_loss(model, batch)
        assert not torch.isnan(loss), "Loss is NaN ngay từ đầu — kiểm tra data/charset"
        assert loss.item() < 20, f"Loss quá cao ({loss.item():.2f}) — kiểm tra label encoding"
        print(f"Sanity check passed — initial loss: {loss.item():.4f}")
    except Exception as e:
        raise RuntimeError(f"Sanity check failed: {e}")
```

---

## 4. Quy tắc inference & production

### 4.1 Luôn export ONNX trước khi ship

Không ship PyTorch `.pt` file cho production. ONNX + ONNX Runtime:
- Không phụ thuộc vào phiên bản PyTorch của user
- Hỗ trợ DirectML (GPU Intel/AMD tích hợp trên Windows)
- Nhanh hơn 2–3× nhờ graph optimization

```python
# Export — chạy một lần
torch.onnx.export(
    model.cpu().eval(), dummy_input, "models/crnn.onnx",
    dynamic_axes={"image": {3: "width"}},  # width phải dynamic
    opset_version=17,
    do_constant_folding=True,
)

# Verify sau export
import onnx
onnx.checker.check_model("models/crnn.onnx")
print("ONNX model OK")
```

### 4.2 Confidence threshold phải configurable

Không hardcode threshold trong code.

```python
# ĐÚNG
class OCRConfig:
    conf_threshold: float = 0.6  # configurable
    min_box_area: int = 100
    max_text_length: int = 200

# SAI
def filter_results(results):
    return [r for r in results if r.conf > 0.6]  # magic number
```

### 4.3 Input validation bắt buộc

```python
def validate_input(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {image.dtype}")
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        raise ValueError(f"Image too small: {w}×{h}")
    if h > 4320 or w > 7680:
        raise ValueError(f"Image too large: {w}×{h} (max 8K)")
```

### 4.4 Thread safety cho inference session

ONNX Runtime session là thread-safe, nhưng preprocessing/postprocessing thì không nhất thiết.

```python
from threading import Lock

class SafeRecognizer:
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path)
        self._lock = Lock()

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        x = self._preprocess(image)   # stateless → thread safe
        with self._lock:
            logits = self.sess.run(None, {"image": x})[0]
        return self._decode(logits)   # stateless → thread safe
```

---

## 5. Quy tắc code

### 5.1 Type hints bắt buộc trên mọi function public

```python
# ĐÚNG
def recognize(self, image: np.ndarray, threshold: float = 0.6) -> tuple[str, float]:
    ...

# SAI
def recognize(self, image, threshold=0.6):
    ...
```

### 5.2 Không dùng global state ngoài logger và config

```python
# ĐÚNG
class Pipeline:
    def __init__(self, config: OCRConfig):
        self.config   = config
        self.detector = DBNetDetector(config.det_model_path)
        self.recognizer = CRNNRecognizer(config.rec_model_path)

# SAI
detector   = DBNetDetector("models/dbnet.onnx")   # global
recognizer = CRNNRecognizer("models/crnn.onnx")   # global

def run(image):
    return recognizer.recognize(detector.detect(image)[0])
```

### 5.3 Mọi model path phải relative, đọc từ config

```python
# ĐÚNG — config file
# configs/default.yaml
# model:
#   det_path: models/dbnet.onnx
#   rec_path: models/crnn.onnx

# SAI — hardcode absolute path
sess = ort.InferenceSession(r"D:\projects\ocr\models\crnn.onnx")
```

### 5.4 Logging thay vì print

```python
import logging

logger = logging.getLogger(__name__)

# ĐÚNG
logger.info(f"Recognized {len(results)} text regions in {elapsed:.1f}ms")
logger.warning(f"Low confidence crop skipped: conf={conf:.2f}")

# SAI
print(f"done: {elapsed}ms")
```

### 5.5 Error handling không nuốt exception im lặng

```python
# ĐÚNG — log và re-raise hoặc trả về có ý nghĩa
def recognize(self, image: np.ndarray) -> tuple[str, float]:
    try:
        return self._run_inference(image)
    except ort.InvalidGraph as e:
        logger.error(f"ONNX graph error: {e}")
        raise
    except Exception as e:
        logger.warning(f"Recognition failed, returning empty: {e}")
        return "", 0.0

# SAI — nuốt exception
def recognize(self, image):
    try:
        return self._run_inference(image)
    except:
        return ""  # bug ẩn, không biết lý do
```

---

## 6. Quy tắc performance

### 6.1 Batch inference khi có thể

Đừng gọi recognize từng crop một khi có nhiều crop trong cùng frame.

```python
# ĐÚNG — batch
def recognize_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
    padded, widths = self._pad_batch(crops)      # pad về cùng width
    logits = self.sess.run(None, {"image": padded})[0]
    return [self._decode(logits[i], widths[i]) for i in range(len(crops))]

# SAI — loop đơn lẻ
results = [recognizer.recognize(crop) for crop in crops]  # chậm hơn ~5×
```

### 6.2 Resize ảnh về H=32 trước khi đưa vào CRNN

CRNN chỉ cần H=32. Đưa ảnh lớn vào là lãng phí.

```python
def preprocess(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_w = max(int(w * 32 / h), 1)
    image = cv2.resize(image, (new_w, 32), interpolation=cv2.INTER_LINEAR)
    # ... normalize
```

### 6.3 Cache font objects khi sinh synthetic data

```python
from functools import lru_cache
from PIL import ImageFont

@lru_cache(maxsize=256)
def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)
# Không gọi ImageFont.truetype(...) trong vòng lặp sinh data
```

### 6.4 Profiling bắt buộc trước khi optimize

```python
import cProfile, pstats

def profile_pipeline(n_frames: int = 100):
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_frames):
        run_ocr_pipeline(test_screenshot)
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    # Xem bottleneck thật sự trước khi đoán mò và optimize
```

---

## 7. Quy tắc kiểm thử

### 7.1 Mỗi module phải có unit test trước khi integrate

```
tests/
├── test_data_gen.py        # Synthetic generation
├── test_charset.py         # Charset encoding/decoding
├── test_preprocess.py      # Image preprocessing
├── test_postprocess.py     # Text cleanup
├── test_recognizer.py      # CRNN inference
├── test_detector.py        # DBNet inference
└── test_pipeline.py        # End-to-end
```

### 7.2 Regression test với hard cases

Tập hợp các case đã từng fail và giữ làm regression suite:

```python
HARD_CASES = [
    ("font_size_8px.png",   "Tiny text"),
    ("cleartype_blue.png",  "ClearType rendering"),
    ("white_on_dark.png",   "Dark mode UI"),
    ("mixed_vn_en.png",     "Tiếng Việt mixed with English"),
    ("code_snippet.png",    "Source code with symbols"),
    ("table_cell.png",      "Text in table border"),
]

def test_hard_cases(recognizer):
    for img_path, expected in HARD_CASES:
        img  = cv2.imread(f"tests/hard_cases/{img_path}")
        pred, conf = recognizer.recognize(img)
        cer  = compute_cer(expected, pred)
        assert cer < 0.05, f"Regression on {img_path}: CER={cer:.3f}, pred='{pred}'"
```

### 7.3 Latency test phải pass trước khi release

```python
import time

def test_latency(pipeline, n=50):
    times = []
    screenshot = take_test_screenshot()
    for _ in range(n):
        t0 = time.perf_counter()
        pipeline.run(screenshot)
        times.append(time.perf_counter() - t0)

    p50 = sorted(times)[n // 2]
    p95 = sorted(times)[int(n * 0.95)]

    assert p50 < 0.080, f"P50 latency {p50*1000:.1f}ms > 80ms"
    assert p95 < 0.150, f"P95 latency {p95*1000:.1f}ms > 150ms"
    print(f"Latency OK — P50: {p50*1000:.1f}ms, P95: {p95*1000:.1f}ms")
```

---

## 8. Anti-patterns — tuyệt đối không làm

| Anti-pattern | Lý do | Thay bằng |
|---|---|---|
| Dùng OCR engine khác làm GT | Circular dependency, học lỗi | Synthetic gen hoặc UIAutomation |
| Hardcode path tuyệt đối | Không chạy được trên máy khác | Config file, relative path |
| Load model mỗi request | Chậm hơn 100× | Singleton, load một lần |
| Nuốt exception im lặng | Bug ẩn, không thể debug | Log + raise hoặc return có ý nghĩa |
| Train không validate trước | Mất hàng giờ rồi mới biết bug | Sanity check 1 batch trước |
| Augment rotation mạnh | Không đúng với screen text | Chỉ blur, noise, brightness |
| Ship `.pt` file | Phụ thuộc PyTorch version | Export ONNX |
| Magic number trong threshold | Không thể tune | Config dataclass |
| Global mutable state | Race condition, khó test | Dependency injection |
| `print()` trong production code | Không kiểm soát được | `logging` với level |

---

## 9. Quy tắc version & release

### 9.1 Semantic versioning cho model

```
models/
├── crnn_v1.0.0.onnx    # Major: kiến trúc thay đổi
├── crnn_v1.1.0.onnx    # Minor: charset thêm ký tự mới
└── crnn_v1.1.1.onnx    # Patch: fine-tune thêm data
```

### 9.2 Metadata bắt buộc kèm mỗi model

```python
MODEL_METADATA = {
    "version":      "1.1.0",
    "charset":      "charset_v2.txt",
    "train_samples": 520000,
    "val_cer":      0.018,
    "val_exact_match": 0.923,
    "created_at":   "2024-06-09",
    "torch_version": "2.2.0",
    "notes":        "Fine-tuned thêm 20k dark mode samples",
}
```

### 9.3 Không xóa model cũ — archive thay vì delete

Khi có model mới, move model cũ vào `models/archive/`. Giữ ít nhất 2 version gần nhất để rollback.

---

## Checklist trước khi merge code

- [ ] Type hints đầy đủ trên function public
- [ ] Unit test đã viết và pass
- [ ] Latency test pass (P50 < 80ms)
- [ ] Không có hardcoded path hoặc magic number
- [ ] Logger thay vì print
- [ ] Input validation có mặt ở entry point
- [ ] ONNX export hoạt động nếu có thay đổi model
- [ ] Metadata model đã cập nhật
- [ ] Hard cases regression test pass
