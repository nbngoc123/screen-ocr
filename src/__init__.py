"""
Screen OCR Engine — src package.

Modules:
    charset    — CharsetCodec, build/load/save charset
    corpus     — Text corpus cho synthetic data generation
    data_gen   — Synthetic image generator
    augment    — Albumentations preprocessing pipeline
    crawler    — UIAutomation screen crawler
    dataset    — PyTorch Dataset + DataLoader
    model      — CRNN architecture (ResNet18 + BiLSTM + CTC)
    train      — Training loop
    detector   — DBNet++ ONNX wrapper (Text Detection)
    recognizer — CRNN ONNX wrapper (Text Recognition)
    postprocess — Text post-processing
    pipeline   — Full end-to-end OCR pipeline
"""
