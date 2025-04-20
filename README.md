# 🎓 Student Performance Classification using Logistic Regression

This project uses logistic regression to predict student performance based on demographic, social, and academic features from the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance).

## 🚀 Overview

The notebook combines:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Encoding and Scaling
- Binary Classification (Pass/Fail)
- Model Training & Evaluation using Logistic Regression

---

## 📊 Dataset Description

The dataset contains attributes like:
- Demographics: age, sex, address
- Family background: parents' education & job
- Academic: study time, failures, absences, grades (G1, G2, G3)
- Lifestyle: alcohol consumption, internet access, etc.

Two datasets were used:
- `student-mat.csv`: Math course data
- `student-por.csv`: Portuguese course data

They were merged into one for a more robust model.

---

## 🔧 Features Engineering

- **Binary Classification**: `G3` (final grade) is converted to binary:
  - `1` if G3 ≥ 8 (Pass)
  - `0` if G3 < 8 (Fail)
- **Encoding**: 
  - Label Encoding for binary categories (e.g., sex, address)
  - One-Hot Encoding for nominal features (e.g., job, guardian)
- **Normalization**: `StandardScaler` applied to numerical columns

---

## 📈 Model Used

- **Logistic Regression** from `sklearn.linear_model`
- Accuracy and classification report used for evaluation

---

## 📉 Sample EDA Visualizations

```python
# Histogram of final grades
plt.hist(df['G3'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Final Grade (G3)')
plt.xlabel('Final Grade')
plt.ylabel('Frequency')
plt.show()
