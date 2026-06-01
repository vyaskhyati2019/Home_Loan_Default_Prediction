# 🏦 House Loan Default Prediction

A deep learning model built with Keras and TensorFlow to predict the likelihood of loan default using historical applicant data.

## 📋 Project Overview

**Course:** Deep Learning with Keras and TensorFlow  
**Domain:** Finance  
**Objective:** Predict whether a loan applicant will repay or default, enabling lenders to make more informed, data-driven credit decisions.

---

## 📊 Dataset

**Source:** Course-provided dataset (not included in this repository due to file size)  
**Records:** 307,511 loan applicants  
**Features:** 122 columns (120 used for modeling)  
**Target:** `TARGET` — 1 = Defaulted, 0 = Repaid  

| Class | Count | Percentage |
|-------|-------|------------|
| Repaid (0) | 282,686 | 91.93% |
| Defaulted (1) | 24,825 | 8.07% |

See `Data_Dictionary.csv` for a full description of all 120 features.

---

## 🔧 Methodology

1. **Load dataset** — 307,511 rows × 122 columns
2. **Null value analysis** — 67 of 122 columns had missing values; imputed with median (numeric) and 'Unknown' (categorical)
3. **TARGET distribution** — identified severe 11.4:1 class imbalance
4. **Class balancing** — applied SMOTE to oversample minority class to 282,686 samples each
5. **Data visualization** — plotted class distribution before and after SMOTE
6. **Encoding** — Label Encoded 16 categorical columns
7. **Model training** — 4-layer Deep Neural Network with BatchNorm, Dropout, and Early Stopping
8. **Evaluation** — Sensitivity (Recall) and AUC-ROC

---

## 🧠 Model Architecture

| Layer | Units | Activation | Regularization |
|-------|-------|------------|----------------|
| Input | 120 features | — | — |
| Hidden Layer 1 | 256 neurons | ReLU | BatchNorm + Dropout (0.4) |
| Hidden Layer 2 | 128 neurons | ReLU | BatchNorm + Dropout (0.3) |
| Hidden Layer 3 | 64 neurons | ReLU | BatchNorm + Dropout (0.2) |
| Hidden Layer 4 | 32 neurons | ReLU | Dropout (0.1) |
| Output | 1 neuron | Sigmoid | — |

**Optimizer:** Adam (lr=0.001) | **Loss:** Binary Crossentropy | **Batch size:** 512

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 92.79% |
| Sensitivity / Recall | 88.96% |
| AUC-ROC Score | 0.9702 |

The model correctly identifies **89 out of every 100 actual defaulters**, with an AUC-ROC of **0.97** — rated Excellent by industry standards.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `Home_Loan_Default_Prediction.ipynb` | Full Colab notebook with all steps and outputs |
| `House_Loan_Project_Report.docx` | Written project report |
| `Data_Dictionary.csv` | Description of all 120 features |
| `README.md` | This file |

---

## 🚀 How to Run

1. Open `Home_Loan_Default_Prediction.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — you will be prompted to upload `loan_data__1_.csv`
4. All steps will execute automatically

---

## 🛠 Libraries Used

- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

