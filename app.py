import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = r"C:\Users\tnusa\OneDrive\Desktop\Credit_Risk_Project"

CSV_PATH = os.path.join(
    BASE_DIR,
    "credit_default.csv"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "creditguard.db"
)


# =====================================================
# DATABASE FUNCTIONS
# =====================================================

def create_database():

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT,
            income REAL,
            age REAL,
            loan REAL,
            loan_to_income REAL,
            probability REAL,
            risk_level TEXT,
            model TEXT,
            assessment_date TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_assessment(
    applicant_name,
    income,
    age,
    loan,
    loan_to_income,
    probability,
    risk_level,
    model
):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO assessments
        (
            applicant_name,
            income,
            age,
            loan,
            loan_to_income,
            probability,
            risk_level,
            model,
            assessment_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            applicant_name,
            income,
            age,
            loan,
            loan_to_income,
            probability,
            risk_level,
            model,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()
    conn.close()


def get_assessments():

    conn = sqlite3.connect(DB_PATH)

    data = pd.read_sql_query(
        """
        SELECT *
        FROM assessments
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return data


create_database()


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(CSV_PATH):

        st.error(
            "Dataset file was not found."
        )

        st.write(
            "Expected location:",
            CSV_PATH
        )

        st.stop()

    data = pd.read_csv(CSV_PATH)

    data.columns = data.columns.str.strip()

    return data


df = load_dataset()


# =====================================================
# REQUIRED COLUMNS
# =====================================================

FEATURES = [
    "Income",
    "Age",
    "Loan",
    "Loan to Income"
]

TARGET = "Default"

required_columns = FEATURES + [TARGET]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Required columns are missing from the dataset."
    )

    st.write(
        "Missing columns:",
        missing_columns
    )

    st.write(
        "Available columns:",
        list(df.columns)
    )

    st.stop()


# =====================================================
# DATA CLEANING
# =====================================================

for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


if df[TARGET].dtype == "object":

    target_mapping = {
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        "Y": 1,
        "N": 0,
        "Default": 1,
        "Non-Default": 0,
        "default": 1,
        "non-default": 0,
        "1": 1,
        "0": 0
    }

    df[TARGET] = df[TARGET].map(
        target_mapping
    )


df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=required_columns
)

df[TARGET] = df[TARGET].astype(int)


# =====================================================
# MODEL DATA
# =====================================================

X = df[FEATURES]

y = df[TARGET]


if y.nunique() < 2:

    st.error(
        "The target column must contain at least two classes."
    )

    st.stop()


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# MODEL CREATION
# =====================================================

logistic_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# =====================================================
# MODEL TRAINING
# =====================================================

logistic_model.fit(
    X_train,
    y_train
)

random_forest_model.fit(
    X_train,
    y_train
)


# =====================================================
# MODEL EVALUATION
# =====================================================

def evaluate_model(model):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    results = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return results


logistic_results = evaluate_model(
    logistic_model
)

random_forest_results = evaluate_model(
    random_forest_model
)


# =====================================================
# BEST MODEL SELECTION
# =====================================================

if (
    random_forest_results["ROC-AUC"]
    >= logistic_results["ROC-AUC"]
):

    best_model = random_forest_model

    best_model_name = "Random Forest"

    best_results = random_forest_results

else:

    best_model = logistic_model

    best_model_name = "Logistic Regression"

    best_results = logistic_results


# =====================================================
# RISK CLASSIFICATION
# =====================================================

def get_risk_level(probability):

    if probability < 0.20:

        return "Very Low Risk"

    elif probability < 0.40:

        return "Low Risk"

    elif probability < 0.60:

        return "Moderate Risk"

    elif probability < 0.80:

        return "High Risk"

    else:

        return "Very High Risk"


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.title("🛡️ CreditGuard AI")

    st.caption(
        "Credit Risk and Loan Decision Support"
    )

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Risk Assessment",
            "What-If Analysis",
            "Assessment History",
            "Model Performance",
            "Dataset Explorer"
        ]
    )

    st.divider()

    st.write("Application Features")

    st.write("• AI-based risk prediction")
    st.write("• Default probability estimation")
    st.write("• Model comparison")
    st.write("• Assessment history")
    st.write("• Dataset analysis")


# =====================================================
# APPLICATION HEADER
# =====================================================

st.title("🛡️ CreditGuard AI")

st.subheader(
    "AI-Powered Credit Risk and Loan Decision Support Platform"
)

st.caption(
    "A machine learning-based prototype for analyzing applicant credit default risk."
)

st.divider()


# =====================================================
# DASHBOARD PAGE
# =====================================================

if selected_page == "Dashboard":

    st.header("Executive Dashboard")

    st.write(
        "Overview of the dataset, applicant risk patterns, and model performance."
    )

    total_records = len(df)

    default_rate = (
        df[TARGET].mean() * 100
    )

    average_income = df["Income"].mean()

    average_loan = df["Loan"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Applicants",
        f"{total_records:,}"
    )

    col2.metric(
        "Default Rate",
        f"{default_rate:.2f}%"
    )

    col3.metric(
        "Average Income",
        f"{average_income:,.2f}"
    )

    col4.metric(
        "Average Loan",
        f"{average_loan:,.2f}"
    )

    st.write("")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader("Default Distribution")

        distribution = (
            df[TARGET]
            .value_counts()
            .sort_index()
        )

        distribution.index = [
            "Non-Default" if value == 0 else "Default"
            for value in distribution.index
        ]

        st.bar_chart(
            distribution
        )

    with chart_col2:

        st.subheader("Income and Loan Relationship")

        scatter_data = df[
            ["Income", "Loan"]
        ].copy()

        scatter_data = scatter_data.set_index(
            "Income"
        )

        st.line_chart(
            scatter_data
        )

    st.divider()

    model_col, history_col = st.columns(
        [1, 2]
    )

    with model_col:

        st.subheader("Selected AI Model")

        st.success(
            best_model_name
        )

        st.metric(
            "ROC-AUC Score",
            f"{best_results['ROC-AUC']:.4f}"
        )

        st.write(
            "Logistic Regression ROC-AUC:",
            f"{logistic_results['ROC-AUC']:.4f}"
        )

        st.write(
            "Random Forest ROC-AUC:",
            f"{random_forest_results['ROC-AUC']:.4f}"
        )

    with history_col:

        st.subheader("Recent Assessments")

        history = get_assessments()

        if history.empty:

            st.info(
                "No applicant assessments have been saved yet."
            )

        else:

            st.dataframe(
                history.head(5),
                use_container_width=True,
                hide_index=True
            )


# =====================================================
# RISK ASSESSMENT PAGE
# =====================================================

elif selected_page == "Risk Assessment":

    st.header("Applicant Risk Assessment")

    st.write(
        "Enter applicant information to estimate the probability of loan default."
    )

    with st.form("risk_assessment_form"):

        applicant_name = st.text_input(
            "Applicant Name"
        )

        col1, col2 = st.columns(2)

        with col1:

            income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

        with col2:

            loan = st.number_input(
                "Requested Loan Amount",
                min_value=0.0,
                value=10000.0,
                step=1000.0
            )

            loan_to_income = st.number_input(
                "Loan to Income Ratio",
                min_value=0.0,
                value=0.20,
                step=0.01
            )

        analyze_button = st.form_submit_button(
            "Analyze Credit Risk",
            use_container_width=True
        )

    if analyze_button:

        input_data = pd.DataFrame(
            {
                "Income": [income],
                "Age": [age],
                "Loan": [loan],
                "Loan to Income": [loan_to_income]
            }
        )

        probability = best_model.predict_proba(
            input_data
        )[0][1]

        risk_level = get_risk_level(
            probability
        )

        st.divider()

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Default Probability",
            f"{probability * 100:.2f}%"
        )

        result_col2.metric(
            "Risk Level",
            risk_level
        )

        result_col3.metric(
            "Selected Model",
            best_model_name
        )

        if probability < 0.40:

            st.success(
                "The applicant has a relatively low predicted default risk."
            )

        elif probability < 0.60:

            st.warning(
                "The applicant has a moderate predicted default risk."
            )

        else:

            st.error(
                "The applicant has a high predicted default risk."
            )

        st.subheader("Risk Factors")

        risk_factors = []

        if loan_to_income > 0.50:

            risk_factors.append(
                "High loan-to-income ratio"
            )

        if loan > income * 0.40:

            risk_factors.append(
                "Requested loan is high compared with income"
            )

        if age < 25:

            risk_factors.append(
                "Applicant is relatively young"
            )

        if income < df["Income"].median():

            risk_factors.append(
                "Income is below the dataset median"
            )

        if not risk_factors:

            risk_factors.append(
                "No major threshold-based risk factor detected"
            )

        for factor in risk_factors:

            st.info(
                factor
            )

        save_assessment(
            applicant_name,
            income,
            age,
            loan,
            loan_to_income,
            probability,
            risk_level,
            best_model_name
        )

        st.success(
            "Assessment saved successfully."
        )

        report_text = f"""
CREDITGUARD AI
Credit Risk Assessment Report
========================================

Applicant Name: {applicant_name}
Annual Income: {income}
Age: {age}
Requested Loan Amount: {loan}
Loan to Income Ratio: {loan_to_income}

Default Probability: {probability * 100:.2f}%
Risk Level: {risk_level}
Selected Model: {best_model_name}

Assessment Date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

========================================
This is an educational machine learning prototype.
It should not be used as the sole basis for actual loan approval.
"""

        st.download_button(
            "Download Assessment Report",
            data=report_text,
            file_name="creditguard_assessment_report.txt",
            mime="text/plain"
        )


# =====================================================
# WHAT-IF ANALYSIS PAGE
# =====================================================

elif selected_page == "What-If Analysis":

    st.header("What-If Credit Risk Analysis")

    st.write(
        "Modify applicant information and observe how the predicted risk changes."
    )

    col1, col2 = st.columns(2)

    with col1:

        what_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            key="what_income"
        )

        what_age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            key="what_age"
        )

    with col2:

        what_loan = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            key="what_loan"
        )

        what_lti = st.number_input(
            "Loan to Income Ratio",
            min_value=0.0,
            value=0.20,
            step=0.01,
            key="what_lti"
        )

    what_if_input = pd.DataFrame(
        {
            "Income": [what_income],
            "Age": [what_age],
            "Loan": [what_loan],
            "Loan to Income": [what_lti]
        }
    )

    what_probability = best_model.predict_proba(
        what_if_input
    )[0][1]

    what_risk = get_risk_level(
        what_probability
    )

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Default Probability",
        f"{what_probability * 100:.2f}%"
    )

    col2.metric(
        "Predicted Risk Level",
        what_risk
    )

    if what_probability < 0.40:

        st.success(
            "Low predicted credit risk"
        )

    elif what_probability < 0.60:

        st.warning(
            "Moderate predicted credit risk"
        )

    else:

        st.error(
            "High predicted credit risk"
        )


# =====================================================
# ASSESSMENT HISTORY PAGE
# =====================================================

elif selected_page == "Assessment History":

    st.header("Assessment History")

    st.write(
        "View previously analyzed applicant records."
    )

    history = get_assessments()

    if history.empty:

        st.info(
            "No assessment history available."
        )

    else:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Assessment History",
            data=history.to_csv(
                index=False
            ),
            file_name="creditguard_assessment_history.csv",
            mime="text/csv"
        )


# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif selected_page == "Model Performance":

    st.header("AI Model Performance")

    st.write(
        "Comparison between Logistic Regression and Random Forest."
    )

    performance_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Logistic Regression": [
                logistic_results["Accuracy"],
                logistic_results["Precision"],
                logistic_results["Recall"],
                logistic_results["F1 Score"],
                logistic_results["ROC-AUC"]
            ],
            "Random Forest": [
                random_forest_results["Accuracy"],
                random_forest_results["Precision"],
                random_forest_results["Recall"],
                random_forest_results["F1 Score"],
                random_forest_results["ROC-AUC"]
            ]
        }
    )

    st.dataframe(
        performance_data.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Best Selected Model: {best_model_name}"
    )

    st.subheader("Confusion Matrix")

    best_predictions = best_model.predict(
        X_test
    )

    matrix = confusion_matrix(
        y_test,
        best_predictions
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.imshow(matrix)

    ax.set_title(
        f"{best_model_name} Confusion Matrix"
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    for row in range(matrix.shape[0]):

        for col in range(matrix.shape[1]):

            ax.text(
                col,
                row,
                matrix[row, col],
                ha="center",
                va="center"
            )

    st.pyplot(
        fig,
        use_container_width=True
    )


# =====================================================
# DATASET EXPLORER PAGE
# =====================================================

elif selected_page == "Dataset Explorer":

    st.header("Dataset Explorer")

    st.write(
        "Explore the dataset used for training the credit risk models."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        len(df)
    )

    col2.metric(
        "Columns",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "CreditGuard AI | Educational Machine Learning Prototype | 2026"
)