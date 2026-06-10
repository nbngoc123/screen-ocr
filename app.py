"""
app.py — Giao diện Web Gradio cho End-to-End OCR (DBNet + CRNN).
"""
import yaml
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from src.pipeline.app import OCRPipeline

# Khởi tạo mô hình lúc start app
with open("configs/default.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("Loading OCR Pipeline (DBNet + CRNN)...")
try:
    pipeline = OCRPipeline(config)
    print("Load mô hình thành công!")
except Exception as e:
    print(f"Lỗi load mô hình: {e}")
    pipeline = None

def draw_results(image_np, results):
    """
    Vẽ bounding box và text đè lên ảnh.
    Sử dụng PIL để hỗ trợ vẽ text tiếng Việt (Unicode).
    """
    # Convert numpy BGR to PIL Image RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Load font tiếng việt cơ bản (dùng font mặc định của PIL hoặc font tải về)
    # Tạm dùng font mặc định nếu không có font truetype
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for r in results:
        b = r.box
        text = r.text
        # Vẽ viền box (Đỏ)
        draw.rectangle([(b.x1, b.y1), (b.x2, b.y2)], outline="red", width=2)
        
        # Lấy kích thước chữ để vẽ nền đen cho dễ đọc
        try:
            bbox = draw.textbbox((b.x1, b.y1 - 25), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = font.getsize(text) if hasattr(font, 'getsize') else (100, 20)
            
        draw.rectangle([(b.x1, max(0, b.y1 - th - 5)), (b.x1 + tw + 4, b.y1)], fill="black")
        draw.text((b.x1 + 2, max(0, b.y1 - th - 5)), text, font=font, fill="white")
        
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def format_markdown(results):
    """Tạo bảng kết quả dạng Markdown"""
    if not results:
        return "*Không nhận diện được chữ nào trong ảnh.*"
        
    md = "| STT | Văn bản nhận diện | Độ tin cậy |\n"
    md += "|---|---|---|\n"
    for i, r in enumerate(results):
        md += f"| {i+1} | **{r.text}** | {r.conf * 100:.2f}% |\n"
    return md

def infer(image):
    if pipeline is None:
        return None, "Lỗi: Pipeline chưa được load (Kiểm tra lại DBNet/CRNN onnx)."
        
    if image is None:
        return None, "Vui lòng tải ảnh lên."
        
    # image mặc định của Gradio (kiểu numpy array) là dạng RGB
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        # Chạy End-to-End Pipeline
        results = pipeline.run(img_bgr)
        
        # Vẽ kết quả
        res_img = draw_results(img_bgr, results)
        
        # Định dạng output text
        res_text = format_markdown(results)
        
        # Gradio image output yêu cầu format là RGB
        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        
        return res_img_rgb, res_text
    except Exception as e:
        return None, f"Lỗi trong quá trình nhận diện: {e}"

# Thiết kế giao diện Gradio
with gr.Blocks(title="Screen OCR - Demo") as demo:
    gr.Markdown("# 🔍 Nhận diện chữ từ ảnh (End-to-End: DBNet + CRNN)")
    gr.Markdown("Tải toàn bộ ảnh màn hình hoặc ảnh lớn vào đây. Mô hình DBNet sẽ dò tìm các khối chữ và cắt ra cho CRNN đọc.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Ảnh đầu vào (Upload nguyên bức ảnh)", type="numpy")
            btn_run = gr.Button("Nhận diện toàn bộ ảnh", variant="primary")
            
        with gr.Column(scale=1):
            out_img = gr.Image(label="Kết quả Box & Text")
            out_text = gr.Markdown(label="Bảng kết quả nhận diện")
            
    # Xử lý sự kiện click
    btn_run.click(
        fn=infer,
        inputs=input_img,
        outputs=[out_img, out_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
