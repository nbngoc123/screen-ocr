# Phiên bản v0.1.0-baseline: Mô hình nhận diện CRNN ban đầu

Đây là phiên bản phát hành đầu tiên (Baseline) của hệ thống Screen OCR, hoàn thành giai đoạn 2 của Roadmap (huấn luyện CRNN Baseline). 

Mô hình hiện đã có khả năng đọc được chữ (recognition) trên các vùng text crop, sau khi được train 86 epochs.

## 📊 Kết quả Đánh giá (Evaluation Metrics)
Dựa trên các báo cáo tự động sinh ra (`reports/`):

- **Trên tập dữ liệu giả lập (Synthetic Test)**
  - Số lượng: 20 mẫu
  - Độ chính xác hoàn toàn (Exact Match): **50.00%**
  - Tỉ lệ lỗi ký tự (CER - Character Error Rate): **9.59%**

- **Trên tập dữ liệu thực tế (ICDAR 2015 Task 3)**
  - Số lượng: 1441 mẫu
  - Độ chính xác hoàn toàn (Exact Match): **47.60%**
  - Tỉ lệ lỗi ký tự (CER): **23.16%**

## 🎯 Đánh giá chung
- Mô hình đã có thể học được các đặc trưng ký tự (Final Train Loss: 0.0769, Val Loss: 0.2630).
- Tuy nhiên, khi áp dụng lên dữ liệu thực tế ICDAR, CER vẫn ở mức khá cao (23%).
- **Bước tiếp theo:** Thu thập thêm dữ liệu thực tế (Real data) và bổ sung các phương pháp Data Augmentation (nhiễu, mờ, bóng...) để fine-tune, nhắm tới mục tiêu đưa CER xuống dưới `< 20%`.
