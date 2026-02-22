# 🍎 Fruit Classification using PyTorch & Flask

A simple image classification project that predicts the type of fruit from an uploaded image using a Convolutional Neural Network (CNN).  
The trained model is deployed as a web application using Flask.

---

## 📌 Project Overview

- Image classification using **PyTorch**
- Custom **CNN** model
- Data augmentation with rotations and mirror flips
- Flask-based web interface
- Beginner-friendly project
- **Test Accuracy: 83%**

---

## 🧠 Model Details

- Architecture: Custom CNN
- Framework: PyTorch
- Input Size: 224 × 224
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Output: Fruit class label

---

## 🔁 Data Augmentation

- Random Rotation (different angles)
- Horizontal Flip (mirror)
- Vertical Flip (mirror)
- Image normalization

---

## 📁 Project Structure

```text
fruit_classification/
│
├── data/
│   ├── train/
│   │   ├── apple/
│   │   ├── banana/
│   │   ├── grape/
│   │   ├── kiwi/
│   │   ├── mango/
│   │   ├── orange/
│   │   ├── pear/
│   │   ├── pineapple/
│   │   ├── strawberry/
│   │   └── watermelon/
│   │
│   └── test/
│       ├── apple/
│       ├── banana/
│       ├── grape/
│       ├── kiwi/
│       ├── mango/
│       ├── orange/
│       ├── pear/
│       ├── pineapple/
│       ├── strawberry/
│       └── watermelon/
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
├── app.py
├── model.py
├── fruit_classifier.pth
├── training.ipynb
├── cleaning.ipynb
├── requirements.txt
└── README.md
