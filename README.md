# resnet50-pneumonia-mnist
Pneumonia Detection Using Transfer Learning (ResNet50)
# Pneumonia Detection Using Transfer Learning (ResNet50)

This project fine-tunes a pre-trained ResNet-50 model to classify chest X-ray images from the [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data) dataset as **pneumonia** or **normal**.

---

## 🎯 Objective

- Detect pneumonia from chest X-ray images using transfer learning.
- Mitigate class imbalance and evaluate the model's performance using robust metrics.

---

## 🗂️ Dataset

**PneumoniaMNIST** (derived from NIH chest X-rays)

- Format: `.npz` containing `train`, `val`, and `test` sets.
- Images are grayscale and resized to 28x28 (expanded to 224x224 for ResNet input).
- Labels: `0 = Normal`, `1 = Pneumonia`

---

## 🔧 Methodology

- **Model**: ResNet-50 (pretrained on ImageNet)
- **Transfer Learning**:
  - Freeze base layers initially
  - Train top classifier head
  - Fine-tune final 40 layers (excluding BatchNorm)
- **Loss**: Focal loss with class weighting to address imbalance
- **Cross-Validation**: 3-fold Stratified K-Fold on training data
- **Final Evaluation**: Best model + Ensemble (mean of CV models)

---

## 📈 Evaluation Metrics

| Metric         | Justification                                                                 |
|----------------|--------------------------------------------------------------------------------|
| Accuracy       | Overall correctness of predictions                                             |
| F1-Score       | Balances precision and recall, crucial for imbalanced datasets                |
| AUC-ROC        | Measures the model's ability to distinguish between the two classes            |
| Confusion Matrix | Offers a detailed breakdown of classification performance                  |

---

## ✅ Results

### 📊 Best CV Model Performance (on Test Set)

Accuracy: 0.7420
F1-Score: 0.8278
AUC-ROC: 0.8903

### 🤝 Ensemble Performance (Averaged CV Models)

Accuracy (Ensemble): 0.7308
F1-Score (Ensemble): 0.8220
AUC-ROC (Ensemble): 0.8906
---

![image](https://github.com/user-attachments/assets/cfd561cf-2cd5-4562-9cb7-0773a8a1cea9)
