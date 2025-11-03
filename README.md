# 🧠 Loan Approval Prediction

This project aims to predict whether a loan application will be **approved or rejected** using machine learning techniques.

---

## 📋 Project Description

In this project, we build a complete data science pipeline to classify loan applications based on applicant and loan features.  
The dataset is preprocessed, imbalanced data is handled using **SMOTE**, and two models are trained and compared:
- **Logistic Regression**
- **Decision Tree Classifier**

The main focus is on improving classification performance using metrics suitable for imbalanced data such as **Precision**, **Recall**, and **F1-score**.

---

## 📊 Dataset

**Source:** Kaggle - [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

**Description:**  
This dataset contains information about applicants and their loan applications,  
used to predict whether a loan will be approved or rejected.

**Attributes:**
- `loan_id` — Unique identifier for each loan application  
- `no_of_dependents` — Number of dependents of the applicant  
- `education` — Applicant’s education level (Graduate / Not Graduate)  
- `self_employed` — Whether the applicant is self-employed (Yes / No)  
- `income_annum` — Annual income of the applicant  
- `loan_amount` — Amount of loan requested  
- `loan_term` — Loan repayment term (in months)  
- `cibil_score` — Applicant’s credit score  
- `residential_assets_value` — Value of residential assets  
- `commercial_assets_value` — Value of commercial assets  
- `luxury_assets_value` — Value of luxury assets (cars, jewelry, etc.)  
- `bank_asset_value` — Value of bank account and financial assets  
- `loan_status` — Target variable (Approved / Rejected)

---

## 🧰 Tools & Libraries

- Python 
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib  
- Seaborn  

---

## ⚙️ Workflow

1. **Data Loading & Cleaning**
   - Load the dataset and remove/handle missing values.  
   - Convert categorical values into numerical form using `OneHotEncoder`.

2. **Data Preprocessing**
   - Handle missing values with `SimpleImputer`.  
   - Scale numeric features using `StandardScaler`.

3. **Handle Imbalanced Data**
   - Apply **SMOTE (Synthetic Minority Oversampling Technique)** to balance the target variable.

4. **Model Training**
   - Train two models: Logistic Regression and Decision Tree.  
   - Use `train_test_split` to divide data into training and testing sets.

5. **Evaluation**
   - Use metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.  
   - Visualize model performance with **ROC Curves** and **Precision–Recall Curves**.

6. **Comparison**
   - Display both models’ performance in a comparison table.

---

## 📊 Results

| Model | Precision | Recall | F1-Score | Accuracy | ROC AUC |
|--------|------------|---------|-----------|-----------|----------|
| Logistic Regression | ~0.90 | ~0.92 | ~0.91 | ~0.92 | ~0.97 |
| Decision Tree | ~0.95 | ~0.97 | ~0.96 | ~0.97 | ~0.99 |

🔹 The **Decision Tree Classifier** achieved slightly better performance overall.  
🔹 Both models handled imbalanced data effectively after applying SMOTE.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/rmmekky/loan-approval-prediction.git
   cd loan-approval-prediction
