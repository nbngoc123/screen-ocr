"""
Gradio UI để test Full OCR Pipeline (DBNet + CRNN)
Cài đặt: pip install gradio
Chạy: python scripts/run_gradio.py
"""
import sys
import os
import cv2
import numpy as np
import gradio as gr

# Thêm thư mục gốc vào path để import src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.inference import OCRInference

print("Đang tải mô hình OCR...")
# Khởi tạo mô hình OCR
ocr_engine = OCRInference()
print("Tải mô hình thành công!")

def process_image(image: np.ndarray):
    """
    Nhận ảnh từ Gradio (numpy array dạng RGB),
    chạy OCR, vẽ Bounding Box và Text lên ảnh.
    """
    if image is None:
        return None, ""
        
    # Chuyển ảnh RGB (Gradio mặc định) sang BGR để encode
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Encode sang byte để gọi predict_image
    _, encoded_img = cv2.imencode('.png', img_bgr)
    image_bytes = encoded_img.tobytes()
    
    # Chạy Inference
    try:
        results = ocr_engine.predict_image(image_bytes)
    except Exception as e:
        return image, f"Lỗi OCR: {e}"
    
    # Vẽ Bounding Box lên ảnh để trả về
    output_image = image.copy() # Vẽ trên ảnh RGB để hiển thị đúng màu trên Web
    output_text = []
    
    for item in results:
        box = item["box"]
        text = item["text"]
        conf = item["confidence"]
        
        x1, y1, x2, y2 = box
        
        # Vẽ Box màu xanh lá
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ Text (OpenCV không hỗ trợ Tiếng Việt tốt, nhưng sẽ vẽ tạm ASCII/chữ không dấu)
        # Để vẽ Tiếng Việt xịn cần PIL, nhưng để đơn giản ta dùng putText hiển thị test
        cv2.putText(
            output_image, text, (x1, max(y1 - 5, 0)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
        )
        
        output_text.append(f"[{x1}, {y1}, {x2}, {y2}] - {text} (conf: {conf:.2f})")
        
    return output_image, "\n".join(output_text)

# Khởi tạo Giao diện Gradio
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Ảnh đầu vào (Upload hoặc Paste)"),
    outputs=[
        gr.Image(label="Ảnh kết quả (Có vẽ Box)"),
        gr.Textbox(label="Kết quả OCR (JSON text)", lines=10)
    ],
    title="Screen OCR Test (DBNet + CRNN)",
    description="Tải lên hình ảnh chụp màn hình để xem hệ thống dò tìm (Detector) và đọc chữ (Recognizer) hoạt động."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
