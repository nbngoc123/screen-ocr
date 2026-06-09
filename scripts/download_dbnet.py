"""
Tải mô hình DBNet++ đã được pre-train và chuyển sang ONNX.
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

def download_model():
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Đang quét danh sách file từ HuggingFace...")
    repos = ["marsena/paddleocr-onnx-models", "monkt/paddleocr-onnx", "Kreuzberg/paddleocr-onnx-models"]
    
    downloaded = False
    for repo in repos:
        try:
            files = list_repo_files(repo_id=repo)
            # Tìm file det (detection) dạng onnx
            det_files = [f for f in files if "det" in f.lower() and f.endswith(".onnx")]
            if det_files:
                target_file = det_files[0]
                print(f"Tìm thấy: {target_file} trong repo {repo}")
                file_path = hf_hub_download(repo_id=repo, filename=target_file, local_dir=str(model_dir))
                
                os.rename(file_path, model_dir / "dbnet.onnx")
                print(f"\n[THÀNH CÔNG] Đã lưu thành: {model_dir / 'dbnet.onnx'}")
                downloaded = True
                break
        except Exception as e:
            print(f"Lỗi truy cập repo {repo}: {e}")
            
    if not downloaded:
        print("[LỖI] Không thể tìm thấy file DBNet ONNX ở bất kỳ repo nào.")

if __name__ == "__main__":
    download_model()
