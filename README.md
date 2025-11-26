# Deep Learning Model Comparison for Image Classification 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red)](https://keras.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## 📖 Overview

This project focuses on evaluating and comparing the performance of various **Deep Learning architectures** (both Custom CNN and Transfer Learning models) on two distinct datasets:
1.  **Vietnamese Cuisine:** Recognizing different types of local food.
2.  **Tourist Landmarks:** Identifying famous travel destinations.

The goal is to analyze accuracy, loss, and training time to determine the most suitable model for mobile or web deployment.

## 📂 Repository Structure

* `Cuisine_Classification.ipynb` (originally `nckh-ma.ipynb`):
    * Notebook for training and evaluating models on the **Food Dataset**.
* `Landmark_Classification.ipynb` (originally `nckh-dd.ipynb`):
    * Notebook for training and evaluating models on the **Landmark/Location Dataset**.

## 🧠 Models Implemented

This project implements and compares the following architectures:

1.  **Custom CNN**: A lightweight Convolutional Neural Network built from scratch.
2.  **MobileNetV2**: Efficient for mobile devices.
3.  **VGG16**: A classic deep architecture.
4.  **ResNet50V2**: Utilizes residual connections for deeper training.
5.  **DenseNet121**: Connects each layer to every other layer in a feed-forward fashion.
6.  **InceptionV3**: Uses factorized convolutions for high performance.

## 🛠️ Tech Stack

* **Core:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Image Augmentation:** ImageDataGenerator (Rescaling, Shear, Zoom, Horizontal Flip)

## 📊 Methodology

For both datasets, the workflow includes:

1.  **Data Preprocessing:**
    * Image resizing to `128x128`.
    * Normalization (rescaling `1./255`).
    * Data augmentation to prevent overfitting.
2.  **Training:**
    * Each model is trained for **15 epochs**.
    * Optimizer: `Adam`.
    * Loss Function: `Categorical Crossentropy`.
3.  **Evaluation:**
    * Comparison based on **Accuracy** and **Loss**.
    * **Confusion Matrix** visualization for detailed error analysis.
    * **Bar Charts** comparing final accuracy across all models.

## 🚀 How to Run

### 1. Prerequisites
Ensure you have the following installed:
```bash
pip install tensorflow pandas numpy matplotlib seaborn
