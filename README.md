# 📡 RF Signal Classification using Machine Learning

<div align="center">

### Automatic Modulation Classification using Random Forest • Support Vector Machine • Residual 1D CNN

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge\&logo=jupyter)


### 🚀 End-to-End Automatic Modulation Classification (AMC) System using Traditional Machine Learning and Deep Learning for RF Signal Recognition.


</div>

---

# 📖 Overview

Automatic Modulation Classification (AMC) is a key component of modern wireless communication systems. It enables intelligent identification of modulation schemes directly from received RF signals without prior information about the transmitter.

This repository presents a complete end-to-end RF Signal Classification pipeline, beginning from dataset preparation and feature engineering to machine learning, deep learning, unknown signal detection, and deployment.

The project combines the strengths of **Random Forest**, **Support Vector Machine (SVM)**, and a **Residual 1D Convolutional Neural Network (CNN)** to classify modulation schemes from IQ samples while supporting confidence-based rejection of unknown signals.

---

# ✨ Project Highlights

| Feature                     | Status |
| --------------------------- | :----: |
| End-to-End Pipeline         |    ✅   |
| Dataset Processing          |    ✅   |
| Feature Engineering         |    ✅   |
| Feature Selection           |    ✅   |
| Random Forest               |    ✅   |
| Support Vector Machine      |    ✅   |
| Residual CNN                |    ✅   |
| Model Comparison            |    ✅   |
| Unknown Signal Detection    |    ✅   |
| Confidence-Based Prediction |    ✅   |
| Offline Deployment          |    ✅   |

---

# 🏗️ Complete Workflow

```text
                    RAW IQ SIGNALS
                          │
                          ▼
                 Dataset Analysis
                          │
                          ▼
                  Data Preprocessing
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
 Feature Extraction                Raw IQ Samples
          │                               │
          ▼                               ▼
 Feature Validation              Residual 1D CNN
          │                               │
          ▼                               ▼
 Feature Selection                  CNN Prediction
          │
     ┌────┴─────┐
     ▼          ▼
Random Forest   SVM
     │          │
     └────┬─────┘
          ▼
 Performance Evaluation
          │
          ▼
 Model Comparison
          │
          ▼
 Unknown Signal Detection
          │
          ▼
 Offline Deployment Showcase
```

---

# 📂 Repository Structure

```text
RF-Signal-Classification
│
├── data
│   ├── raw
│   ├── processed
│   └── samples
│
├── notebooks
│   ├── Dataset Analysis
│   ├── Preprocessing
│   ├── Feature Extraction
│   ├── Dataset Cleaning
│   ├── Feature Validation
│   ├── Feature Selection
│   ├── Random Forest
│   ├── Support Vector Machine
│   ├── CNN Dataset Preparation
│   ├── CNN Training
│   ├── CNN Evaluation
│   ├── Performance Analysis
│   └── Unknown Signal Detection
│
├── models
│
├── results
│
├── demo
│   └── showcase.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Supported Modulation Schemes

| Digital Modulations | Analog Modulations |
| ------------------- | ------------------ |
| BPSK                | AM-DSB             |
| QPSK                | AM-SSB             |
| 8PSK                |                    |
| PAM4                |                    |
| GFSK                |                    |
| CPFSK               |                    |
| QAM64               |                    |

---

# 🤖 Machine Learning & Deep Learning Models

| Model                     | Description                                                                   |
| ------------------------- | ----------------------------------------------------------------------------- |
| 🌳 Random Forest          | Ensemble-based classifier using handcrafted statistical and spectral features |
| 📈 Support Vector Machine | Margin-based classifier trained on engineered RF features                     |
| 🧠 Residual 1D CNN        | End-to-end deep learning model that learns directly from raw IQ samples       |

---

# ⚙️ Technologies Used

| Category             | Tools            |
| -------------------- | ---------------- |
| Programming Language | Python           |
| Machine Learning     | Scikit-Learn     |
| Deep Learning        | PyTorch          |
| Data Processing      | NumPy, Pandas    |
| Visualization        | Matplotlib       |
| Notebook Environment | Jupyter Notebook |
| Model Serialization  | Joblib, Pickle   |

---

# 📈 Evaluation Metrics

The trained models are evaluated using:

* ✅ Accuracy
* ✅ Precision
* ✅ Recall
* ✅ F1 Score
* ✅ Confusion Matrix
* ✅ Per-Class Accuracy
* ✅ Accuracy vs SNR
* ✅ Inference Latency
* ✅ Model Size
* ✅ Confidence Distribution

---

# 🚨 Unknown Signal Detection

Instead of forcing every received signal into one of the trained modulation classes, the system performs confidence-based rejection.

```text
             Prediction
                  │
                  ▼
        Calculate Confidence
                  │
                  ▼
     Confidence ≥ Threshold ?
          │                │
          │                │
         YES              NO
          │                │
          ▼                ▼
Known Modulation     Unknown Signal
```

This makes the deployment more robust and reliable in real-world RF environments.

---

# 🖥️ Deployment Demonstration

The repository includes a complete demonstration program (`showcase.py`) that performs:

* Loading trained models
* Random RF signal selection
* Random Forest prediction
* SVM prediction
* CNN prediction
* Confidence comparison
* Majority voting
* Unknown signal detection
* Final prediction summary

---

# 📈 Project Statistics

| Metric                  | Value |
| ----------------------- | ----- |
| Machine Learning Models | 2     |
| Deep Learning Models    | 1     |
| Supported Modulations   | 9     |
| Feature Engineering     | ✔     |
| Unknown Detection       | ✔     |
| Offline Deployment      | ✔     |
| Dataset Processing      | ✔     |

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/<your-username>/RF-Signal-Classification.git

cd RF-Signal-Classification
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Random Forest

```bash
python train_random_forest.py
```

---

## Train SVM

```bash
python train_svm.py
```

---

## Train CNN

```bash
python train_cnn.py
```

---

## Run Deployment Showcase

```bash
python demo/showcase.py
```

---

# 🎯 Applications

* 📡 Automatic Modulation Classification
* 📶 Cognitive Radio
* 🛰️ Spectrum Monitoring
* 🛡️ Electronic Warfare
* 🔒 Wireless Security
* 📻 Software Defined Radio (SDR)
* 📈 Signal Intelligence
* 📚 Wireless Communication Research

---

# 🛣️ Future Work

* [x] Random Forest Classifier
* [x] Support Vector Machine
* [x] Residual CNN
* [x] Feature Engineering
* [x] Unknown Signal Detection
* [x] Offline Deployment
* [ ] Transformer-Based Models
* [ ] Real-Time SDR Integration
* [ ] GNU Radio Integration
* [ ] FPGA Deployment
* [ ] Edge AI Deployment
* [ ] ONNX Export

---

# 👨‍💻 Authors

### **Shaad Ali**

Electronics & Communication Engineering
KIET Group of Institutions

### **Yugratna**

Electronics & Communication Engineering
KIET Group of Institutions

### **Suhani Pareek**

Electronics & Communication Engineering
KIET Group of Institutions

---

# 🙏 Acknowledgements

Special thanks to:

* Defence Research and Development Organisation (DRDO)
* RadioML Dataset
* Scikit-Learn
* PyTorch
* The Open Source Community

---

<div align="center">

## ⭐ Star this repository if you found it useful!

**Built with ❤️ using Python, Scikit-Learn and PyTorch**

</div>
