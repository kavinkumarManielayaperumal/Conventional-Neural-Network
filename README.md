# 📌 CIFAR-10 Image Classification with CNN (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** using PyTorch to classify images from the **CIFAR-10 dataset**. The model is trained to recognize **10 different classes** including airplanes, cars, birds, cats, and more.

> **Note:** This project was created **for fun and understanding CNNs**, which is why the model achieves **67% accuracy**. Feel free to **clone** the repository and modify it to improve the model's accuracy!

## 🚀 Features
- ✅ **CNN-based image classification model** using PyTorch
- ✅ **Training and evaluation scripts**
- ✅ **Batch loading & pre-processing using DataLoader**
- ✅ **Model checkpointing and saving (`cnn_model.pth`)**
- ✅ **Performance evaluation with accuracy & F1-score**

---

## 📂 Project Structure
```
📦 Conventional-Neural-Network
 ┣ 📜 train.py             # Training script
 ┣ 📜 evaluate.py          # Model evaluation
 ┣ 📜 data_loader.py       # Loads and batches dataset
 ┣ 📜 dataset_viewer.py    # Previews dataset images
 ┣ 📜 Model.py             # CNN architecture
 ┣ 📜 cnn_model.pth        # Saved trained model
 ┣ 📜 predictions.txt      # Model predictions output
 ┣ 📜 README.md            # This file (Project documentation)
 ┣ 📜 requirements.txt     # Python dependencies
 ┣ 📜 Dockerfile           # Containerization setup
 ┗ 📂 dataset              # CIFAR-10 dataset files
```

---

## 📦 Installation & Setup
1️⃣ **Clone this repository:**
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt
```

3️⃣ **Run Training:**
```bash
python train.py
```

4️⃣ **Run Evaluation:**
```bash
python evaluate.py
```

---

## 🏋️‍♂️ Training Details
- **Dataset**: CIFAR-10
- **Model**: CNN with 4 Convolutional Layers
- **Optimizer**: Adam (`lr=0.001`)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 10

---

## 🎯 Results
| Metric        | Value |
|--------------|-------|
| **Accuracy**  | 67.36% |
| **F1 Score**  | 0.67  |
| **Precision** | 0.68  |

📌 **Classification Report:** (F1-score per class)
```
Class 0 (Airplane): Precision = 0.76, Recall = 0.72, F1-score = 0.74
Class 1 (Automobile): Precision = 0.84, Recall = 0.76, F1-score = 0.80
Class 5 (Dog): Precision = 0.47, Recall = 0.43, F1-score = 0.45
```

---

## 📈 Future Improvements
- 🔹 **Data Augmentation** (Flip, Rotate, Crop)
- 🔹 **Train deeper networks** (ResNet, VGG)
- 🔹 **Hyperparameter tuning** (Optimizer, Batch size)
- 🔹 **Use Transfer Learning** for better accuracy

---

## 👨‍💻 Contributors
- **Your Name** ([GitHub Profile](https://github.com/your-username))
- **Other Contributors**

---

## ⭐ Support
If you found this helpful, please consider **starring** 🌟 the repo!
