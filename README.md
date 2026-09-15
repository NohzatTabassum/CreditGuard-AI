# CreditGuard AI 🏦🤖

An AI-powered Credit Risk & Loan Decision Support Platform built with Python, Machine Learning, Streamlit, and SQLite.

## 📌 Project Overview

CreditGuard AI is a machine learning-based application designed to assess the credit risk of loan applicants. It predicts the probability of loan default and provides a simple risk classification to support loan decision-making.

The system uses applicant information such as income, age, loan amount, and loan-to-income ratio to estimate credit risk.

> **Note:** This project is an educational/prototype decision-support system and is not intended for real-world banking approval decisions.

## 🚀 Features

* 📊 Interactive Dashboard
* 🏦 Credit Risk Assessment
* 🤖 Machine Learning-based Default Prediction
* 🔍 What-If Analysis
* 📈 Model Performance Analysis
* 🗂️ Assessment History
* 📁 Dataset Explorer
* 💾 SQLite Database Integration
* 📄 Downloadable Assessment Report
* ⚡ Automatic Best Model Selection

## 🧠 Machine Learning Models

Two machine learning algorithms are implemented:

* Logistic Regression
* Random Forest Classifier

The application automatically compares the models using **ROC-AUC** and selects the better-performing model for prediction.

## 📊 Risk Classification

The predicted default probability is converted into risk levels:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

This provides an easy-to-understand interpretation of the model's prediction.

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* SQLite
* Machine Learning

## 📂 Dataset

The project uses a Credit Default dataset containing information such as:

* Income
* Age
* Loan Amount
* Loan-to-Income Ratio
* Default Status

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/NohzatTabassum/CreditGuard-AI.git
```

### 2. Open the project folder

```bash
cd CreditGuard-AI
```

### 3. Install required libraries

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## 📁 Project Structure

```text
CreditGuard-AI/
│
├── app.py
├── credit_default.csv
├── creditguard.db
└── README.md
```

## 🔮 Future Improvements

* Integration of more financial features
* Explainable AI for individual predictions
* Advanced credit scoring models
* Real-time financial data integration
* User authentication and role-based access
* Deployment on a cloud platform

## 👩‍💻 Author

**Nohzat Tabassum**

Software Engineering Student
Daffodil International University

## ⭐ Project Purpose

This project was developed as a practical machine learning and software engineering project to demonstrate the application of AI in financial risk assessment and decision-support systems.
