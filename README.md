# Deepfake Image Detection using MobileNetV2 and ResNet18

## Project Overview
Deepfake technology, powered by Generative Adversarial Networks (GANs), has enabled the creation of hyper-realistic synthetic images.
While these advancements open creative opportunities, they also pose serious risks such as misinformation, identity theft, and security threats.  

This project presents a **deepfake image detection system** using **MobileNetV2** and **ResNet18** architectures.
Both models are optimized with transfer learning to classify **real vs. fake facial images** with high accuracy while remaining lightweight for potential real-time deployment.

---

## Objectives
- Develop an efficient deep learning model to classify **real vs. fake images**.
- Leverage **MobileNetV2** and **ResNet18** with transfer learning.
- Optimize preprocessing pipelines for dataset uniformity and model generalization.
- Evaluate performance using **accuracy, precision, recall, F1-score, and confusion matrices**.
- Demonstrate feasibility for deployment in **mobile and edge devices**.

---

## Tech Stack
- **Language:** Python 3.7+
- **Frameworks:** PyTorch, Torchvision, TensorFlow/Keras (optional)
- **Libraries:** NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn, Seaborn
- **Environment:** Jupyter Notebook / Google Colab / VS Code
- **Hardware (Colab):**
  - GPU: NVIDIA Tesla T4 (16GB VRAM)  
  - RAM: 12.7 GB  

---

##  Dataset
- Source: [Kaggle – Real and Fake Face Detection Dataset](https://www.kaggle.com/datasets/uditsharma72/real-vs-fake-faces)
- **Classes:**  
  - `real/` → Authentic human faces  
  - `fake/` → GAN-generated deepfake faces  
- **Split:**  
  - 70% training (1400 images)  
  - 15% validation (300 images)  
  - 15% testing (300 images)  

Images are resized to **224x224** and normalized using ImageNet statistics. Data augmentation techniques (rotation, flips, zoom) were applied to improve generalization.

---

##  Methodology

### 1. Preprocessing
- Resize →  224x224 pixels  
- Normalize → Using ImageNet mean & std  
- Augmentaxtion → Flips, rotations, zoom  

### 2. Models
- **MobileNetV2**: Lightweight CNN optimized with depthwise separable convolutions.  
- **ResNet18**: Residual learning to enable deeper networks without vanishing gradients.  

### 3. Training
- Optimizer: **Adam** (lr=0.0001)  
- Loss: **CrossEntropyLoss**  
- Epochs: **10**  
- Batch size: **32**  
- Mixed Precision Training enabled for efficiency.  

### 4. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## Results

### MobileNetV2
- **Accuracy:** 76.22%  
- **Precision:** 74.13%  
- **Recall:** 74.65%  
- **F1-score:** 74.39%  

### ResNet18
- **Accuracy:** 67%  
- **Precision:** 71%  
- **Recall:** 51%  
- **F1-score:** 59%  

 **Observation:**  
- MobileNetV2 outperformed ResNet18 in both accuracy and recall.  
- ResNet18 showed better precision but lower recall.  
- Both models indicated potential overfitting on validation data.  

---

## Future Work
- Use **larger and more diverse datasets**.  
- Explore **Vision Transformers (ViTs)** and **EfficientNet**.  
- Build an **ensemble of models** for robust predictions.  
- Integrate **Explainable AI (Grad-CAM)** for interpretability.  
- Deploy real-time detection via **TensorFlow Lite** or **ONNX**.  

---
