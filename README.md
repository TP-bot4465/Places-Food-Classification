# Deep Learning for Vietnamese Food & Landmark Classification 🇻🇳📸

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red)](https://keras.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## 📖 Giới thiệu (Overview)

Repository này chứa source code thực nghiệm nhằm đánh giá và so sánh hiệu năng của các kiến trúc **Deep Learning** (Transfer Learning) trên bài toán **Image Classification** với bối cảnh đặc trưng tại Việt Nam.

Dự án thực hiện Train và Evaluate trên 2 bộ dữ liệu (Datasets) riêng biệt:
1.  **Vietnamese Food Dataset:** Nhận diện **21 loại món ăn** phổ biến (Bánh xèo, Phở, Bún bò, v.v.).
2.  **Vietnam Landmarks Dataset:** Nhận diện **26 địa danh/địa điểm** du lịch nổi tiếng (Bitexco, Landmark 81, Chợ Bến Thành, v.v.).

Mục tiêu chính là phân tích các chỉ số **Accuracy**, **Loss** và **F1-Score** để tìm ra mô hình tối ưu nhất.

## 📂 Cấu trúc Repository

* `foodsClassification.ipynb`:
    * Notebook huấn luyện và đánh giá trên tập dữ liệu món ăn (21 classes).
* `placesClassification.ipynb`:
    * Notebook huấn luyện và đánh giá trên tập dữ liệu địa danh (26 classes).

## 🧠 Các mô hình được thử nghiệm (Models)

Dự án sử dụng kỹ thuật **Transfer Learning** với bộ trọng số `imagenet` trên 5 kiến trúc mạng nơ-ron tiên tiến:

1.  **InceptionV3**
2.  **Xception**
3.  **MobileNetV2** (Tối ưu cho thiết bị di động/Edge devices)
4.  **ResNet152V2** (Kiến trúc rất sâu với Residual connections)
5.  **InceptionResNetV2** (Kết hợp Inception module và Residual connections)

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas, OpenCV (cv2), PIL
* **Visualization:** Matplotlib
* **Metrics:** Scikit-learn (F1 Score, Confusion Matrix)

## 📊 Phương pháp thực hiện (Methodology)

Quy trình thực nghiệm (Pipeline) được áp dụng thống nhất cho cả 2 bài toán:

### 1. Data Preprocessing (Tiền xử lý)
* **Input Shape:** Resize toàn bộ ảnh về kích thước `299x299`.
* **Normalization:** Rescaling pixel values về khoảng `[0, 1]` (`1.0/255`).
* **Data Generator:** Sử dụng `ImageDataGenerator` để load dữ liệu theo batch.

### 2. Model Training (Huấn luyện)
* **Fine-tuning:** Đóng băng các layer của Base Model (`trainable = False`), thêm các lớp fully connected mới:
    * `Conv2D` + `MaxPooling2D` (cho Food dataset) hoặc `GlobalAveragePooling2D` (cho Landmark dataset).
    * `Dense` (Relu/Softmax).
    * `Dropout` (0.2) để giảm Overfitting.
* **Hyperparameters:**
    * **Epochs:** 30
    * **Optimizer:** Adam (`learning_rate=1e-4`)
    * **Loss Function:** Categorical Crossentropy
    * **Callbacks:** `EarlyStopping` (patience=3), `ModelCheckpoint` (lưu model tốt nhất).

### 3. Evaluation (Đánh giá)
* Đánh giá mô hình dựa trên các chỉ số: **Training/Validation Accuracy** và **Loss**.
* Tính toán **Weighted F1-Score** cho từng class.
* Trực quan hóa kết quả bằng **Confusion Matrix** để phân tích các trường hợp nhận diện sai (misclassification).

## 🚀 Hướng dẫn cài đặt & Sử dụng (How to Run)

### 1. Cài đặt thư viện (Prerequisites)
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn opencv-python
