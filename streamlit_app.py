import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Wine Quality ML Classifier Evaluator",
    page_icon="🍷",
    layout="wide",
)

st.title("🍷 Machine Learning Classifier Evaluation Dashboard")
st.markdown("### BITS Pilani M.Tech (DSE) - Assignment 2")

# Sidebar Configuration
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Support Vector Machine",
    ],
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Test Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    if "target" not in data.columns:
        st.error("Error: Ground truth column 'target' not found in uploaded file.")
    else:
        X_test = data.drop(columns=["target"])
        y_test = data["target"]

        model_filename = f"model_{model_choice.lower().replace(' ', '_')}.joblib"
        try:
            model = joblib.load(model_filename)
            scaler = joblib.load("scaler.joblib")

            # Scaled model evaluation
            if model_choice in ["Logistic Regression", "kNN", "Support Vector Machine"]:
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test

            y_pred = model.predict(X_eval)
            y_proba = (
                model.predict_proba(X_eval)[:, 1]
                if hasattr(model, "predict_proba")
                else np.zeros(len(y_pred))
            )

            # Compute Evaluation Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.markdown(f"## 📊 Model Performance: **{model_choice}**")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("AUC Score", f"{auc:.4f}")
            m3.metric("Precision", f"{prec:.4f}")
            m4.metric("Recall", f"{rec:.4f}")
            m5.metric("F1 Score", f"{f1:.4f}")
            m6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")

            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    ax=ax,
                    xticklabels=["Low Quality (0)", "High Quality (1)"],
                    yticklabels=["Low Quality (0)", "High Quality (1)"],
                )
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                st.pyplot(fig)

            with col_b:
                st.subheader("📝 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

        except FileNotFoundError as e:
            st.error(f"Missing required model artifact file: `{model_filename}`. Make sure all `.joblib` files are uploaded to GitHub.")
        except Exception as e:
            st.error(f"Execution Error: {e}")
else:
    st.info("👈 Please upload the `test_data.csv` file using the sidebar to run the evaluation.")