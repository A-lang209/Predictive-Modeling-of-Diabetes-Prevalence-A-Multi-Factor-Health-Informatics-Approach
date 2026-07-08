# Predictive-Modeling-of-Diabetes-Prevalence-A-Multi-Factor-Health-Informatics-Approach

📊 Automated Diabetes Risk Prediction Pipeline Using Random Forest

This repository contains a production-ready, highly optimized Python pipeline designed for Google Colab to explore, visualize, and predict diabetes diagnoses from multi-dimensional patient clinical data.

🚀 Project Overview

The pipeline processes structured health records for 1,879 patients across 40+ attributes—including demographics, lifestyle factors, medical history, clinical measurements, and environmental exposures—to predict binary classification status (Diagnosis).

🛠️ Architecture & Workflow

Data Ingestion & Fast-Track Execution: Optimized for minimal runtime overhead. Utilizes a direct drag-and-drop local file system mapping (/content/diabetes_data.csv) to eliminate interactive execution lags.

Feature Engineering & Pre-processing: Automated handling of high-variance features utilizing StandardScaler to achieve zero mean and unit variance, protecting downstream models from scale bias.

Imbalanced Ensemble Learning: Implements a RandomForestClassifier initialized with 100 decision trees (n_estimators=100) and stratified training splits (stratify=y). Out-of-the-box support for medical class imbalances via cost-sensitive learning (class_weight='balanced').

Diagnostic Metrics Suite: Fully integrated evaluation utilizing:

1) Confusion Matrix Heatmaps to isolate and minimize False Negatives.
2) ROC-AUC Curves to evaluate true positive/false positive rate trade-offs.
3) MDI Feature Importance Extraction to surface the top 15 predictive drivers.

📈 Dependencies

1) pandas
2) numpy
3) scikit-learn
4) matplotlib
5) seaborn
