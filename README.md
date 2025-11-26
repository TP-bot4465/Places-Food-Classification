# Comparative Analysis of Deep Learning Models for Image Classification 📸

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red)](https://keras.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## 📖 Giới thiệu (Overview)

Repository này chứa source code thực nghiệm nhằm đánh giá và so sánh hiệu năng của các kiến trúc **Deep Learning** (bao gồm Custom CNN và các mô hình Transfer Learning phổ biến) trên bài toán **Image Classification**.

Dự án thực hiện train và test trên 2 bộ dữ liệu (Datasets) riêng biệt:
1.  **Cuisine Dataset:** Nhận diện các món ăn đặc trưng.
2.  **Landmark Dataset:** Nhận diện các địa điểm/địa danh du lịch.

Mục tiêu chính là phân tích các chỉ số **Accuracy**, **Loss** và **Training Time** để tìm ra mô hình tối ưu nhất cho việc triển khai thực tế.

## 📂 Cấu trúc Repository

* `Cuisine_Classification.ipynb` (Code gốc: `nckh-ma.ipynb`):
    * Notebook dùng để huấn luyện và đánh giá trên tập dữ liệu món ăn.
* `Landmark_Classification.ipynb` (Code gốc: `nckh-dd.ipynb`):
    * Notebook dùng để huấn luyện và đánh giá trên tập dữ liệu địa danh.

## 🧠 Các mô hình được thử nghiệm (Models)

Dự án triển khai và so sánh 6 kiến trúc mạng nơ-ron khác nhau:

1.  **Custom CNN**: Một mạng Convolutional Neural Network cơ bản được xây dựng từ đầu (scratch).
2.  **MobileNetV2**: Kiến trúc tối ưu cho thiết bị di động (Mobile/Edge devices).
3.  **VGG16**: Kiến trúc CNN cổ điển với độ sâu lớn.
4.  **ResNet50V2**: Sử dụng Residual Connections để giải quyết vấn đề vanishing gradient.
5.  **DenseNet121**: Kết nối các layer theo kiểu feed-forward dày đặc (Dense connectivity).
6.  **InceptionV3**: Sử dụng Inception modules để tăng hiệu quả tính toán.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Data Augmentation:** ImageDataGenerator (Rescaling, Shear, Zoom, Horizontal Flip)

## 📊 Phương pháp thực hiện (Methodology)

Quy trình thực nghiệm (Pipeline) cho cả 2 bộ dữ liệu bao gồm:

1.  **Data Preprocessing (Tiền xử lý):**
    * Resize ảnh về kích thước `128x128`.
    * **Normalization**: Rescaling pixel values về khoảng `[0, 1]`.
    * **Data Augmentation**: Áp dụng các kỹ thuật biến đổi ảnh để giảm thiểu Overfitting.
2.  **Model Training (Huấn luyện):**
    * Mỗi mô hình được train trong **15 Epochs**.
    * **Optimizer**: Adam.
    * **Loss Function**: Categorical Crossentropy.
3.  **Evaluation (Đánh giá):**
    * So sánh dựa trên **Training/Validation Accuracy** và **Loss**.
    * Trực quan hóa kết quả bằng **Confusion Matrix** và biểu đồ cột (Bar Charts).

## 🚀 Hướng dẫn cài đặt & Sử dụng (How to Run)

### 1. Cài đặt thư viện (Prerequisites)
Đảm bảo môi trường Python của bạn đã cài đặt các thư viện cần thiết:
```bash
pip install tensorflow pandas numpy matplotlib seaborn
