# 🖌️ Kanji - Nhận diện chữ Kanji viết tay (Handwritten Kanji Recognition)

**Kanji** là ứng dụng sử dụng Trí tuệ nhân tạo (Deep Learning) để nhận diện chữ Hán (Kanji) tiếng Nhật viết tay theo thời gian thực. Dự án sử dụng mô hình **EfficientNetB0** và **Simple CNN** được huấn luyện trên dữ liệu sinh từ nhiều font chữ khác nhau.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## ✨ Tính năng chính

*   **Nhận diện chính xác:** Hỗ trợ nhận diện **1006 chữ Kanji** (Bộ Kyōiku Kanji - Giáo dục tiểu học).
*   **Hai chế độ Model:**
    *   🧠 **EfficientNetB0:** Độ chính xác cao , phù hợp máy mạnh.
    *   ⚡ **Simple CNN:** Siêu nhẹ, tốc độ phản hồi tức thì, phù hợp máy cấu hình thấp.
*   **Giao diện vẽ thông minh (GUI):**
    *   Bảng vẽ mượt mà.
    *   **Smart Crop:** Tự động cắt, căn giữa và phóng to nét vẽ để tăng độ chính xác.
    *   **Invert Color:** Chế độ đảo màu nền (Đen/Trắng) linh hoạt.
*   **Hỗ trợ Debug:** Xem trực tiếp hình ảnh mà AI "nhìn thấy" để tinh chỉnh cách vẽ.

## 📂 Cấu trúc thư mục

```text
Kanji-Project/
│
├── test.py                     # File chạy chương trình chính (App vẽ)
├── best_efficientnet_kanji.h5  # File Model EfficientNet (Download từ Kaggle)
├── best_simple_cnn.h5          # File Model Simple CNN (Download từ Kaggle)
├── kanji_labels_map.pkl        # File từ điển ánh xạ (Số -> Chữ Kanji)
├── kanji_labels_map_1.pkl      # (Tùy chọn) File từ điển cho Simple CNN nếu train riêng
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Tài liệu hướng dẫn

## Cài đặt các thư viện cần thiết:
Mở terminal (CMD/PowerShell) và chạy lệnh:

```text
 pip install tensorflow numpy pillow opencv-python scikit-learn

## 🚀 Hướng dẫn sử dụng
**1. Chạy ứng dụng**
Chạy file test.py bằng Python:
```text
python test.py

**2. Cách dùng trên giao diện**
Vẽ chữ: Dùng chuột vẽ chữ Kanji vào khung màu đen.
Lưu ý: Nên vẽ nét dứt khoát, to và rõ ràng.
Nhận diện: Thả chuột ra, kết quả sẽ hiện ngay bên dưới kèm theo độ tin cậy (%).
Xóa bảng: Nhấn nút Xóa bảng (hoặc Clear) để vẽ chữ mới.
Nếu máy đoán sai:
Thử tích vào ô [x] Đảo màu (nếu model được train với nền trắng chữ đen).
Kiểm tra xem bạn đã vẽ đúng nét chưa.
**3. Chuyển đổi Model**
Mở file test.py bằng trình soạn thảo code, tìm dòng CẤU HÌNH ở đầu file và bỏ comment model bạn muốn dùng:
```text
# Chọn 1 trong 2 dòng dưới đây:
MODEL_PATH = "best_efficientnet_kanji.h5"  # Dùng EfficientNet (Khuyên dùng)
# MODEL_PATH = "best_simple_cnn.h5"        # Dùng Simple CNN
