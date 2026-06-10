import os
import sys
import yaml
import cv2
import csv
import json
import torch
import pytest
import argparse
import jiwer
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset.charset import CharsetCodec
from src.recognizer.model import CRNN, ctc_greedy_decode
from src.data_generation.augment import preprocess_for_crnn

def load_recognizer(config_path="configs/default.yaml", weight_path="checkpoints/crnn-v2.pth"):
    """
    Hàm tiện ích để load model CRNN đã train.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load cấu hình
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    codec = CharsetCodec(config["paths"]["charset"])
    
    # Khởi tạo kiến trúc
    model = CRNN(
        num_classes=len(codec),
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=0.0
    )
    
    # Load weights nếu file tồn tại
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model_state = model.state_dict()
        
        # Tự động thay đổi kích thước layer fc nếu charset bị thay đổi
        if "fc.weight" in state_dict and "fc.weight" in model_state:
            ckpt_shape = state_dict["fc.weight"].shape
            model_shape = model_state["fc.weight"].shape
            if ckpt_shape != model_shape:
                print(f"[CẢNH BÁO] Kích thước fc.weight không khớp: Checkpoint {ckpt_shape} vs Model {model_shape}. Đang tự động map kích thước...")
                new_weight = model_state["fc.weight"].clone()
                min_out = min(ckpt_shape[0], model_shape[0])
                min_in = min(ckpt_shape[1], model_shape[1])
                new_weight[:min_out, :min_in] = state_dict["fc.weight"][:min_out, :min_in]
                state_dict["fc.weight"] = new_weight

        if "fc.bias" in state_dict and "fc.bias" in model_state:
            ckpt_shape = state_dict["fc.bias"].shape
            model_shape = model_state["fc.bias"].shape
            if ckpt_shape != model_shape:
                print(f"[CẢNH BÁO] Kích thước fc.bias không khớp: Checkpoint {ckpt_shape} vs Model {model_shape}. Đang tự động map kích thước...")
                new_bias = model_state["fc.bias"].clone()
                min_out = min(ckpt_shape[0], model_shape[0])
                new_bias[:min_out] = state_dict["fc.bias"][:min_out]
                state_dict["fc.bias"] = new_bias
                
        # Dùng strict=False để bỏ qua các lỗi nhỏ nếu cấu trúc có lệch nhẹ
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    
    return model, codec, config, device

def predict_text(img_path, model, codec, config, device):
    """
    Dự đoán văn bản từ một đường dẫn ảnh cụ thể.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
        
    target_h = config["preprocess"]["target_h"]
    
    # Tiền xử lý: Resize chiều cao về target_h, giữ nguyên tỷ lệ chiều rộng
    img_tensor_np = preprocess_for_crnn(img, target_h=target_h, is_train=False)
    
    # (H, W) -> (1, 1, H, W)
    img_tensor = torch.from_numpy(img_tensor_np).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        preds = ctc_greedy_decode(outputs, codec.charset)
        
    return preds[0]

def evaluate_dataset(model, codec, config, device, gt_path, img_dir, out_csv=None):
    """
    Đánh giá độ chính xác (Accuracy, CER, WER) trên tập test có nhãn (GT).
    File GT có định dạng: filename.png, "label"
    """
    if not os.path.exists(gt_path) or not os.path.exists(img_dir):
        print(f"Không tìm thấy dữ liệu test: {gt_path} hoặc {img_dir}")
        return
        
    predictions = []
    targets = []
    correct = 0
    total = 0
    report_data = []  # Lưu dữ liệu để xuất CSV
    
    with open(gt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"\n--- ĐANG ĐÁNH GIÁ TRÊN {len(lines)} MẪU TỪ TẬP TEST ---")
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        if gt_path.endswith(".jsonl"):
            # Format jsonl của generator (run_gen.py)
            data = json.loads(line)
            filename = data.get("file", data.get("img", ""))
            label = data.get("label", "")
            if not filename:
                continue
        else:
            # Format csv/txt cũ của ICDAR (word_1.png, "bada")
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            filename = parts[0].strip()
            label = parts[1].strip().strip('"')
            
        # Trong JSONL của gendata, filename có thể là đường dẫn tương đối (00_000000.png) 
        # nên ta có thể nối trực tiếp. Nếu filename chứa cả đường dẫn thì Path(img_dir)/Path(filename).name
        img_path = Path(img_dir) / Path(filename).name
        if not img_path.exists():
            continue
            
        pred_text = predict_text(img_path, model, codec, config, device)
        
        # Thống kê
        is_correct = (pred_text == label)
        if is_correct:
            correct += 1
            
        item_cer = 0.0
        # Thêm vào list để tính CER/WER (bỏ qua chuỗi rỗng)
        if len(label.strip()) > 0:
            targets.append(label)
            predictions.append(pred_text if len(pred_text.strip()) > 0 else " ")
            try:
                item_cer = jiwer.cer([label], [pred_text if len(pred_text.strip()) > 0 else " "])
            except:
                pass
                
        report_data.append([filename, label, pred_text, is_correct, round(item_cer * 100, 2)])
        total += 1
        
        if (idx + 1) % 100 == 0:
            print(f" Đã xử lý {idx + 1}/{len(lines)} mẫu...")
            
    # Tính toán các chỉ số
    accuracy = (correct / total) * 100 if total > 0 else 0
    try:
        cer = jiwer.cer(targets, predictions) * 100
        wer = jiwer.wer(targets, predictions) * 100
    except:
        cer, wer = 100.0, 100.0
        
    print(f"\n===== KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ =====")
    print(f"Tổng số mẫu hợp lệ: {total}")
    print(f"Đoán đúng hoàn toàn (Exact Match): {correct}/{total} ({accuracy:.2f}%)")
    print(f"CER (Character Error Rate): {cer:.2f}%")
    print(f"WER (Word Error Rate):      {wer:.2f}%")
    print("=====================================")
    
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        # Xuất CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "target", "prediction", "is_correct", "cer(%)"])
            writer.writerows(report_data)
        
        # Xuất file HTML để xem ảnh dễ dàng
        out_html = out_csv.replace(".csv", ".html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Báo cáo Đánh giá CRNN</title>")
            f.write("<style>")
            f.write("table { border-collapse: collapse; width: 100%; font-family: sans-serif; }")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            f.write("th { background-color: #f2f2f2; }")
            f.write(".incorrect { background-color: #ffe6e6; }")
            f.write("img { max-height: 40px; }")
            f.write("</style></head><body>")
            f.write("<h2>Báo cáo Đánh giá Nhận dạng (CRNN)</h2>")
            f.write(f"<p><strong>Đoán đúng:</strong> {correct}/{total} ({accuracy:.2f}%) | <strong>CER:</strong> {cer:.2f}% | <strong>WER:</strong> {wer:.2f}%</p>")
            f.write("<table><tr><th>Ảnh</th><th>Tên File</th><th>Thực tế (Target)</th><th>Dự đoán (Prediction)</th><th>CER (%)</th></tr>")
            
            # Ưu tiên hiển thị những ảnh đoán sai lên đầu
            sorted_report = sorted(report_data, key=lambda x: x[3])
            
            for row in sorted_report:
                fname, target, pred, is_correct, row_cer = row
                img_path_rel = f"../{img_dir}/{fname}"
                tr_class = "" if is_correct else " class='incorrect'"
                f.write(f"<tr{tr_class}>")
                f.write(f"<td><img src='{img_path_rel}' alt='{fname}' loading='lazy'></td>")
                f.write(f"<td>{fname}</td>")
                f.write(f"<td>{target}</td>")
                f.write(f"<td>{pred}</td>")
                f.write(f"<td>{row_cer}</td>")
                f.write("</tr>\n")
                
            f.write("</table></body></html>")
            
        print(f"[*] Đã xuất báo cáo CSV tại: {out_csv}")
        print(f"[*] Đã xuất báo cáo HTML (kèm ảnh) tại: {out_html}")
    
    return accuracy, cer, wer

# ==============================================================================
# PyTest Cases
# ==============================================================================

def test_recognizer_init():
    """Đảm bảo khởi tạo mô hình thành công và load được Charset"""
    model, codec, _, _ = load_recognizer()
    assert model is not None, "Khởi tạo model thất bại"
    assert len(codec.charset) > 0, "Không load được charset"

@pytest.mark.skipif(not os.path.exists("checkpoints/crnn.pth"), reason="Chưa có file weight đã train")
def test_inference_on_sample_dir():
    """Kiểm tra nhận diện thử trên một vài ảnh raw để đánh giá nhanh"""
    model, codec, config, device = load_recognizer()
    
    # Chỉ định một thư mục có ảnh để test (ví dụ: data/raw/Recognition2)
    test_dir = Path("data/raw/Recognition2")
    if not test_dir.exists():
        pytest.skip(f"Không tìm thấy thư mục {test_dir} để test")
        
    images = list(test_dir.glob("*.png"))[:10]  # Thử nghiệm trên 10 ảnh đầu tiên
    
    print("\n--- KẾT QUẢ SUY LUẬN TRÊN TẬP RAW ---")
    for img_path in images:
        text = predict_text(img_path, model, codec, config, device)
        print(f"[+] {img_path.name}: '{text}'")
        assert isinstance(text, str), "Kết quả nhận dạng phải là chuỗi (string)"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình CRNN")
    parser.add_argument("--gt", type=str, default="data/raw/Test_Task3_GT.txt", help="Đường dẫn đến file nhãn (.txt hoặc .jsonl)")
    parser.add_argument("--img-dir", type=str, default="data/raw/Test_Task3_Images", help="Thư mục chứa ảnh test")
    parser.add_argument("--out", type=str, default="reports/evaluation_report_v2.csv", help="Đường dẫn file CSV xuất ra")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="File cấu hình")
    parser.add_argument("--weight", type=str, default="checkpoints/crnn-v2.pth", help="File trọng số .pth")
    args = parser.parse_args()

    model, codec, config, device = load_recognizer(config_path=args.config, weight_path=args.weight)
    
    gt_file = args.gt
    img_folder = args.img_dir
    out_csv = args.out
    
    if os.path.exists(gt_file) and os.path.exists(img_folder):
        evaluate_dataset(model, codec, config, device, gt_file, img_folder, out_csv)
    else:
        print(f"Lỗi: Không tìm thấy GT '{gt_file}' hoặc Thư mục '{img_folder}'")
