Mình nghĩ nghi ngờ của bạn hoàn toàn hợp lý. Thực tế trong OCR tiếng Việt, lỗi:

```text
o ↔ 0
I ↔ l
1 ↔ l
ă ↔ â
a ↔ á
```

thường đến từ **feature extractor + dữ liệu**, chứ không phải CTC.

Nhìn vào code CRNN của bạn, có vài điểm đáng chú ý.

---

## 1. Vấn đề lớn nhất: ResNet18 ImageNet không sinh ra để đọc chữ

Bạn đang dùng:

```python
backbone = resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
)
```

ImageNet được train để phân biệt:

```text
dog
cat
car
airplane
```

không phải:

```text
a
à
á
ả
ã
ạ
ă
â
```

---

Ví dụ:

```text
a
á
```

khác nhau chủ yếu ở:

```text
dấu sắc
```

chỉ vài pixel.

Trong khi feature của ResNet18 pretrained thường quan tâm:

```text
cạnh
texture
hình dạng lớn
```

---

Kết quả:

```text
á
à
ả
ã
ạ
```

dễ bị embedding gần nhau.

---

## 2. Chiều cao 32 là khá thấp với tiếng Việt

Bạn đang dùng:

```python
(B,1,32,W)
```

Sau đó:

```python
conv1 stride=2
maxpool stride=2
```

---

Ngay từ đầu:

```text
32
↓
16
↓
8
```

---

Một dấu sắc:

```text
á
```

có thể chỉ còn:

```text
1-2 pixel
```

ở feature map.

---

Đây là lý do OCR tiếng Việt hay mất dấu.

---

## 3. Bạn đang squeeze chiều cao thành 1

Cuối CNN:

```python
feat = self.cnn(x)
```

mong đợi:

```text
(B,512,1,W')
```

---

Sau đó:

```python
feat.squeeze(2)
```

---

Điều này nghĩa là toàn bộ thông tin:

```text
dấu trên
dấu dưới
```

đã bị nén vào:

```text
1 hàng pixel
```

---

Đối với:

```text
a
á
à
ả
ã
ạ
```

sự khác biệt nằm chính xác ở:

```text
chiều dọc
```

---

Đây là điểm mình nghi ngờ nhất.

---

## 4. Bạn chưa có attention

Pipeline hiện tại:

```text
Image
↓
ResNet18
↓
BiLSTM
↓
CTC
```

---

Khi gặp:

```text
l
I
1
```

model chỉ nhìn local feature.

---

Không có:

```text
Attention
Transformer Decoder
Language Model
```

để sửa lỗi.

---

## 5. Greedy decode cũng làm lỗi nặng hơn

Bạn đang:

```python
_, preds = logits.max(2)
```

---

Đây là:

```text
Greedy Decode
```

---

Nó chọn:

```text
xác suất cao nhất
```

ở từng timestep.

---

Ví dụ:

```text
á : 0.45
à : 0.42
a : 0.40
```

Greedy:

```text
á
```

---

Nhưng toàn chuỗi có thể:

```text
à
```

mới là hợp lý.

---

Beam Search thường giảm CER thêm:

```text
10-30%
```

so với greedy.

---

# Mình nghi nhất chỗ nào?

Nếu phải xếp hạng:

### 1️⃣ Dữ liệu

```text
60%
```

Nếu dataset thiếu:

```text
ă
â
ê
ô
ơ
ư
```

hoặc mất cân bằng dấu.

---

### 2️⃣ Height = 32

```text
20%
```

---

### 3️⃣ ResNet18 pretrained

```text
10%
```

---

### 4️⃣ Greedy decode

```text
5%
```

---

### 5️⃣ CTC

```text
5%
```

---

## Cách kiểm tra nhanh

In confusion matrix.

Ví dụ:

```text
GT      Pred

á       a
á       a
á       a
à       a
ă       a
â       a
```

---

Nếu thấy:

```text
mọi ký tự có dấu
↓
a
```

thì gần như chắc chắn:

```text
feature extractor đang làm mất thông tin dấu
```

chứ không phải lỗi decode.

Mình khá muốn xem thêm phần **preprocess ảnh trước khi đưa vào CRNN** (resize, crop text box, normalize, giữ aspect ratio thế nào). Trong OCR tiếng Việt, lỗi mất dấu thường xuất hiện từ preprocess còn nhiều hơn từ model.

Mình nghĩ nghi ngờ của bạn hoàn toàn hợp lý. Thực tế trong OCR tiếng Việt, lỗi:

```text
o ↔ 0
I ↔ l
1 ↔ l
ă ↔ â
a ↔ á
```

thường đến từ **feature extractor + dữ liệu**, chứ không phải CTC.

Nhìn vào code CRNN của bạn, có vài điểm đáng chú ý.

---

## 1. Vấn đề lớn nhất: ResNet18 ImageNet không sinh ra để đọc chữ

Bạn đang dùng:

```python
backbone = resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
)
```

ImageNet được train để phân biệt:

```text
dog
cat
car
airplane
```

không phải:

```text
a
à
á
ả
ã
ạ
ă
â
```

---

Ví dụ:

```text
a
á
```

khác nhau chủ yếu ở:

```text
dấu sắc
```

chỉ vài pixel.

Trong khi feature của ResNet18 pretrained thường quan tâm:

```text
cạnh
texture
hình dạng lớn
```

---

Kết quả:

```text
á
à
ả
ã
ạ
```

dễ bị embedding gần nhau.

---

## 2. Chiều cao 32 là khá thấp với tiếng Việt

Bạn đang dùng:

```python
(B,1,32,W)
```

Sau đó:

```python
conv1 stride=2
maxpool stride=2
```

---

Ngay từ đầu:

```text
32
↓
16
↓
8
```

---

Một dấu sắc:

```text
á
```

có thể chỉ còn:

```text
1-2 pixel
```

ở feature map.

---

Đây là lý do OCR tiếng Việt hay mất dấu.

---

## 3. Bạn đang squeeze chiều cao thành 1

Cuối CNN:

```python
feat = self.cnn(x)
```

mong đợi:

```text
(B,512,1,W')
```

---

Sau đó:

```python
feat.squeeze(2)
```

---

Điều này nghĩa là toàn bộ thông tin:

```text
dấu trên
dấu dưới
```

đã bị nén vào:

```text
1 hàng pixel
```

---

Đối với:

```text
a
á
à
ả
ã
ạ
```

sự khác biệt nằm chính xác ở:

```text
chiều dọc
```

---

Đây là điểm mình nghi ngờ nhất.

---

## 4. Bạn chưa có attention

Pipeline hiện tại:

```text
Image
↓
ResNet18
↓
BiLSTM
↓
CTC
```

---

Khi gặp:

```text
l
I
1
```

model chỉ nhìn local feature.

---

Không có:

```text
Attention
Transformer Decoder
Language Model
```

để sửa lỗi.

---

## 5. Greedy decode cũng làm lỗi nặng hơn

Bạn đang:

```python
_, preds = logits.max(2)
```

---

Đây là:

```text
Greedy Decode
```

---

Nó chọn:

```text
xác suất cao nhất
```

ở từng timestep.

---

Ví dụ:

```text
á : 0.45
à : 0.42
a : 0.40
```

Greedy:

```text
á
```

---

Nhưng toàn chuỗi có thể:

```text
à
```

mới là hợp lý.

---

Beam Search thường giảm CER thêm:

```text
10-30%
```

so với greedy.

---

# Mình nghi nhất chỗ nào?

Nếu phải xếp hạng:

### 1️⃣ Dữ liệu

```text
60%
```

Nếu dataset thiếu:

```text
ă
â
ê
ô
ơ
ư
```

hoặc mất cân bằng dấu.

---

### 2️⃣ Height = 32

```text
20%
```

---

### 3️⃣ ResNet18 pretrained

```text
10%
```

---

### 4️⃣ Greedy decode

```text
5%
```

---

### 5️⃣ CTC

```text
5%
```

---

## Cách kiểm tra nhanh

In confusion matrix.

Ví dụ:

```text
GT      Pred

á       a
á       a
á       a
à       a
ă       a
â       a
```

---

Nếu thấy:

```text
mọi ký tự có dấu
↓
a
```

thì gần như chắc chắn:

```text
feature extractor đang làm mất thông tin dấu
```

chứ không phải lỗi decode.

Mình khá muốn xem thêm phần **preprocess ảnh trước khi đưa vào CRNN** (resize, crop text box, normalize, giữ aspect ratio thế nào). Trong OCR tiếng Việt, lỗi mất dấu thường xuất hiện từ preprocess còn nhiều hơn từ model.
Nếu mục tiêu là giảm lỗi:

```text
a ↔ á
a ↔ à
ă ↔ â
o ↔ 0
I ↔ l
1 ↔ l
```

thì mình sẽ ưu tiên chỉnh theo thứ tự sau.

# 1. Chiều cao ảnh OCR (QUAN TRỌNG NHẤT)

Hiện tại:

```yaml
preprocess:
  target_h: 32
```

Thử:

```yaml
preprocess:
  target_h: 48
```

hoặc:

```yaml
preprocess:
  target_h: 64
```

---

Tiếng Việt có:

```text
á à ả ã ạ
ă â
ê ô ơ ư
```

Dấu nằm ở:

```text
trên
dưới
```

ký tự.

Nếu height:

```text
32px
```

thì sau CNN dấu rất dễ biến mất.

---

OCR tiếng Anh:

```text
32px
```

ổn.

OCR tiếng Việt:

```text
48-64px
```

thường tốt hơn rõ rệt.

---

# 2. Crop rộng hơn

Detector của bạn:

```yaml
detector:
  unclip_ratio: 2.0
```

---

Nếu box cắt sát:

```text
á
```

có thể thành:

```text
a
```

---

Thử:

```yaml
detector:
  unclip_ratio: 2.5
```

hoặc:

```yaml
detector:
  min_padding_y: 6
```

---

Đặc biệt:

```text
padding theo chiều dọc
```

rất quan trọng.

---

# 3. Tăng dữ liệu có dấu

Hiện config:

```yaml
data_gen:
  n_train: 20000
```

20k vẫn khá nhỏ.

---

Nên:

```yaml
n_train: 100000
```

hoặc hơn.

---

Đồng thời thống kê:

```python
Counter(all_characters)
```

xem:

```text
á
à
ả
ã
ạ
ă
â
```

xuất hiện bao nhiêu lần.

---

Nếu:

```text
a : 50000

á : 300
à : 200
â : 100
```

thì model sẽ học:

```text
cứ đoán a
```

---

# 4. Giảm augmentation phá dấu

Nếu đang dùng:

```python
GaussianBlur
MotionBlur
Resize mạnh
```

quá mức.

---

Ví dụ:

```text
á
```

blur mạnh:

```text
a
```

---

Đây là lỗi rất phổ biến.

---

# 5. Beam Search

Hiện:

```python
ctc_greedy_decode()
```

---

Thử:

```python
ctcdecode
```

hoặc:

```python
pyctcdecode
```

---

Thường giảm:

```text
CER 10-30%
```

mà không cần train lại.

---

# 6. Chỉnh backbone

Hiện:

```python
ResNet18
```

---

Có thể thử:

```python
ResNet34
```

hoặc:

```python
EfficientNet
```

---

Nhưng đây KHÔNG phải ưu tiên đầu tiên.

---

# 7. Charset

Kiểm tra:

```python
len(charset)
```

và đảm bảo có đủ:

```text
à á ả ã ạ
ă ắ ằ ẳ ẵ ặ
â ấ ầ ẩ ẫ ậ

è é ẻ ẽ ẹ
ê ế ề ể ễ ệ

ì í ỉ ĩ ị

ò ó ỏ õ ọ
ô ố ồ ổ ỗ ộ
ơ ớ ờ ở ỡ ợ

ù ú ủ ũ ụ
ư ứ ừ ử ữ ự

ỳ ý ỷ ỹ ỵ

đ
```

---

Nếu thiếu:

```text
ệ
```

thì model không thể predict được:

```text
ệ
```

dù train bao lâu.

---

# 8. Kiểm tra confusion matrix

Thay vì nhìn loss.

Ví dụ:

```text
GT     Pred

á      a
á      a
à      a
â      a
ă      a
```

---

Nếu xảy ra như trên:

```text
feature extractor hoặc dataset
```

có vấn đề.

---

# Nếu là dự án OCR màn hình Windows của bạn

Mình sẽ thử theo đúng thứ tự này:

### Bước 1

```yaml
target_h: 48
```

---

### Bước 2

```yaml
n_train: 100000
```

---

### Bước 3

Tăng:

```yaml
min_padding_y: 6
```

---

### Bước 4

Thêm:

```python
CER
Confusion Matrix
```

để xem model đang nhầm ký tự nào.

---

### Bước 5

Đổi:

```python
ctc_greedy_decode
```

→

```python
beam search decode
```

Thông thường chỉ riêng các bước 1–5 đã giảm đáng kể lỗi mất dấu và nhầm ký tự trong OCR tiếng Việt mà chưa cần thay đổi kiến trúc CRNN.
