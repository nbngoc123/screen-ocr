import io
import sys
from pathlib import Path
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.inference import OCRInference

# Khởi tạo mô hình
print("Đang nạp mô hình OCR...")
ocr_engine = OCRInference()
print("Nạp mô hình thành công!")

def process_image(image: Image.Image):
    if image is None:
        return None
        
    # Chuyển PIL Image sang bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()
    
    # Thực hiện dự đoán
    results = ocr_engine.predict_image(image_bytes)
    
    # Tạo bản sao của ảnh để vẽ (tránh thay đổi ảnh gốc nếu cần)
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Cố gắng load một font hỗ trợ Unicode/Tiếng Việt, nếu không có thì dùng default
    try:
        # Tùy hệ điều hành mà font này có thể khác nhau (ví dụ: arial.ttf trên windows)
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for res in results:
        box = res['box'] # [x1, y1, x2, y2]
        text = res['text']
        
        # Vẽ bounding box (khung màu xanh)
        draw.rectangle(box, outline="green", width=3)
        
        # Vẽ text ngay phía trên bbox
        # textbbox trả về (left, top, right, bottom)
        text_bbox = draw.textbbox((box[0], max(0, box[1] - 25)), text, font=font)
        
        # Vẽ nền đen cho chữ để dễ nhìn
        draw.rectangle(text_bbox, fill="black")
        
        # Vẽ chữ trắng
        draw.text((box[0], max(0, box[1] - 25)), text, font=font, fill="white")
        
    return annotated_img

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload ảnh màn hình hoặc văn bản"),
    outputs=gr.Image(type="pil", label="Kết quả nhận dạng"),
    title="Screen OCR Test (ONNX)",
    description="Giao diện test nhanh mô hình OCR (DBNet + CRNN) đã được convert sang ONNX. Upload một ảnh bất kỳ để xem mô hình detect và đọc chữ nhé!"
)

if __name__ == "__main__":
    # Chạy ở port 7860
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
