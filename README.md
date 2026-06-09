# Screen OCR Engine

Hệ thống OCR tự build cho màn hình Windows — nhận diện text realtime từ screenshot, phục vụ automation, RAG, dịch thuật, game modding.

## Kiến trúc

```
[mss capture] → [DBNet++ detect] → [crop ROIs] → [CRNN recognize] → [post-process] → [text output]
```

| Thành phần | Công nghệ | Ghi chú |
|---|---|---|
| Screen Capture | `mss` | Layer 1 |
| Preprocessing | OpenCV + albumentations | Layer 2 |
| Text Detection | DBNet++ (ONNX) | Layer 3 — TODO: chưa có model |
| Text Recognition | CRNN custom (ONNX) | Layer 3 — train từ đầu |
| Output | FastAPI + JSON | Layer 4 |

## Cấu trúc thư mục

```
screen-ocr/
├── data/             ← Dataset (gitignored)
├── models/           ← ONNX weights (gitignored)
├── src/              ← Source code chính
├── api/              ← FastAPI service
├── scripts/          ← Utility scripts
├── configs/          ← Config files
├── tests/            ← Unit tests
├── notebooks/        ← Jupyter notebooks (EDA, experiments)
└── reports/          ← Evaluation outputs
```

## Setup

### 1. Tạo conda environment

```bash
conda env create -f environment.yml
conda activate screen-ocr
```

### 2. Tạo charset và liệt kê fonts

```bash
python scripts/list_fonts.py
python -c "from src.charset import save_charset; save_charset()"
```

## Roadmap

| Tuần | Milestone |
|---|---|
| 1 | Data pipeline: synthetic gen + UIAutomation crawler |
| 2 | Train CRNN baseline, CER < 10% |
| 3 | Fine-tune với real data, CER < 2% |
| 4 | Export ONNX, FastAPI, end-to-end < 100ms |
| 5+ | Tích hợp DBNet++, downstream tasks |

## Quy tắc quan trọng

Xem [screen-ocr-rules.md](../screen-ocr-rules.md) để biết đầy đủ các nguyên tắc bắt buộc.

- Detection và Recognition là **2 module tách biệt**
- Model chỉ load **một lần** (Singleton)
- Không dùng OCR engine khác làm **ground truth**
- Dùng **ONNX** cho production (không ship `.pt`)
- Mọi config trong `configs/default.yaml`, không hardcode
