# Machine Learning Assignment 2 - BITS Pilani M.Tech (DSE)

## a. Problem Statement
The goal of this project is to implement, evaluate, and deploy multiple Machine Learning classification algorithms to predict red wine quality based on chemical properties. The project involves end-to-end model training, metric evaluation, interactive application design using Streamlit, and deployment on Streamlit Community Cloud.

## b. Dataset Description
* **Dataset Name:** UCI Red Wine Quality Dataset
* **Source:** UCI Machine Learning Repository
* **Instances:** 1,599 samples
* **Features:** 12 features (Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol, Total Acidity [Engineered])
* **Target Variable:** Binary Classification (`target`): High Quality ($\ge 6$, Class 1) vs. Low Quality ($< 6$, Class 0).

## c. GitHub Repository Link
* **GitHub Repository:** [INSERT_YOUR_GITHUB_REPO_URL_HERE]
* **Live Streamlit App:** [INSERT_YOUR_STREAMLIT_APP_URL_HERE]

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7406 | 0.8240 | 0.7683 | 0.7368 | 0.7522 | 0.4808 |
| **Decision Tree** | 0.7469 | 0.7450 | 0.7586 | 0.7719 | 0.7652 | 0.4908 |
| **kNN** | 0.7250 | 0.8071 | 0.7456 | 0.7368 | 0.7412 | 0.4479 |
| **Naive Bayes** | 0.7125 | 0.7852 | 0.7762 | 0.6491 | 0.7070 | 0.4358 |
| **Support Vector Machine** | 0.7594 | 0.8373 | 0.8013 | 0.7310 | 0.7645 | 0.5219 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed well as a linear baseline (AUC 0.8240), providing consistent balance across precision and recall on standardized features. |
| **Decision Tree** | Achieved high recall (0.7719) and good accuracy (0.7469), but registered a lower AUC (0.7450) due to uncalibrated probability thresholds. |
| **kNN** | Showed moderate predictive capability (F1 0.7412) but was sensitive to local distance variations despite feature scaling. |
| **Naive Bayes** | High precision (0.7762) but lowest recall (0.6491) due to strong conditional independence assumptions across chemical features. |
| **Support Vector Machine** | Delivered top performance with highest Accuracy (0.7594), AUC (0.8373), Precision (0.8013), and MCC (0.5219). |
| **Overall Winner** | **Support Vector Machine (SVM)** is the best performing model overall for this dataset across key metrics (AUC, Precision, and MCC). |