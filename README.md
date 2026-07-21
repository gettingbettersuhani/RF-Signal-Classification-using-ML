# 📡 RF Signal Classification using Machine Learning

<div align="center">

### Automatic Modulation Classification using Random Forest • Support Vector Machine • Residual 1D CNN

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge\&logo=jupyter)

---

*A complete end-to-end Automatic Modulation Classification (AMC) system capable of identifying RF modulation schemes from raw IQ signals using both traditional Machine Learning and Deep Learning techniques.*

</div>

---

# 📖 Overview

Automatic Modulation Classification (AMC) is an essential task in modern wireless communication, cognitive radio, electronic warfare, and spectrum monitoring.

This project develops a complete RF Signal Classification pipeline that:

- Generates and processes RF IQ datasets
- Extracts handcrafted statistical and spectral features
- Trains Random Forest and Support Vector Machine classifiers
- Implements a Residual 1D Convolutional Neural Network for end-to-end learning
- Performs comprehensive performance evaluation
- Supports Unknown Signal Detection using confidence thresholds
- Demonstrates deployment through an offline DRDO showcase application

---

# ✨ Key Features

✅ End-to-End RF Signal Classification Pipeline

✅ Supports Traditional ML and Deep Learning

✅ Random Forest Classifier

✅ Support Vector Machine (SVM)

✅ Residual 1D CNN

✅ Feature Engineering & Selection

✅ Automatic Dataset Preparation

✅ Unknown Signal Detection

✅ Confidence-Based Classification

✅ Performance Comparison

✅ Air-Gapped Offline Deployment Ready

---

# 📂 Project Structure

```text
RF-Signal-Classification/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── notebooks/
│   ├── Dataset Analysis
│   ├── Preprocessing
│   ├── Feature Extraction
│   ├── Feature Validation
│   ├── Feature Selection
│   ├── Random Forest
│   ├── SVM
│   ├── CNN Dataset Preparation
│   ├── CNN Training
│   ├── CNN Evaluation
│   ├── Performance Analysis
│   └── Unknown Signal Detection
│
├── src/
│   ├── data/
│   ├── features/
│   ├── machine_learning/
│   ├── deep_learning/
│   └── utils/
│
├── models/
│
├── results/
│
├── demo/
│   └── showcase.py
│
├── requirements.txt
└── README.md
```

---

# 🔬 Workflow

```text
IQ Signals
      │
      ▼
Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Feature Validation
      │
      ▼
Feature Selection
      │
      ├───────────────┐
      ▼               ▼
Random Forest       SVM
      │               │
      └──────┬────────┘
             │
             ▼
Performance Evaluation

────────────────────────────

Raw IQ Signals
      │
      ▼
Residual 1D CNN
      │
      ▼
Prediction

────────────────────────────

RF + SVM + CNN
      │
      ▼
Performance Analysis
      │
      ▼
Unknown Signal Detection
      │
      ▼
Offline Deployment
```

---

# 📊 Supported Modulation Classes

- BPSK
- QPSK
- 8PSK
- PAM4
- GFSK
- CPFSK
- AM-DSB
- AM-SSB
- QAM64

---

# 🧠 Machine Learning Models

## Random Forest

- Handcrafted Feature Based
- Fast Training
- High Interpretability
- Excellent Baseline Performance

---

## Support Vector Machine

- Handcrafted Feature Based
- Effective Decision Boundaries
- Strong Generalization

---

## Residual 1D CNN

- End-to-End Deep Learning
- Learns Features Automatically
- Skip Connections (Residual Blocks)
- Superior Feature Representation

---

# 📈 Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Per-Class Accuracy
- Accuracy vs SNR
- Inference Latency
- Model Size
- Confidence Distribution

---

# 🚀 Unknown Signal Detection

Instead of forcing every signal into a known modulation class, the system supports confidence-based rejection.

```text
Prediction
      │
      ▼
Confidence Score
      │
      ▼
Confidence ≥ Threshold ?

      │ Yes
      ▼
Known Modulation

      │ No
      ▼
Unknown Signal
```

This improves reliability during deployment.

---

# 🖥️ Deployment Showcase

The repository includes an interactive demonstration application.

The showcase:

- Loads trained models
- Randomly selects RF signals
- Performs prediction using all three models
- Displays confidence scores
- Performs majority voting
- Detects unknown signals
- Generates visualization reports

---

# ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- PyTorch
- Matplotlib
- Joblib
- Jupyter Notebook

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/yourusername/RF-Signal-Classification.git

cd RF-Signal-Classification
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Models

### Random Forest

```bash
python train_random_forest.py
```

### SVM

```bash
python train_svm.py
```

### CNN

```bash
python train_cnn.py
```

---

## Run Showcase

```bash
python demo/showcase.py
```

---

# 🎯 Applications

- Cognitive Radio
- Spectrum Monitoring
- Electronic Warfare
- Wireless Security
- Signal Intelligence
- SDR Systems
- Communication Research

---

# 📚 Future Work

- Transformer-based AMC Models
- Real-Time SDR Integration
- GNU Radio Integration
- FPGA Deployment
- ONNX Model Export
- Quantized Edge Deployment
- Additional Modulation Schemes
- Live RF Signal Acquisition

---

# 👨‍💻 Authors

**Shaad Ali**

Electronics & Communication Engineering

KIET Group of Institutions

---

**Suhani Pareek**

Electronics & Communication Engineering

KIET Group of Institutions

---

**Yugratna**

Electronics & Communication Engineering

KIET Group of Institutions

---

# 🙏 Acknowledgements

- Defence Research and Development Organisation (DRDO)
- RadioML Dataset
- Scikit-Learn
- PyTorch
- Open Source Community

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
