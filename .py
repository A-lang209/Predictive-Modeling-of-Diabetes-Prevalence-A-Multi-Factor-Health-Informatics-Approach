import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Set aesthetic styling for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Direct, instant file path reading
file_path = '/content/diabetes_data.csv'

# Check if the file exists before attempting to read it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    print("--- Dataset Loaded Instantly ---")
    print(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
else:
    print(f"Error: The file '{file_path}' was not found. Please ensure it is uploaded or the path is correct.")



# ==========================================
# 2. DATA EXPLORATION & CLEANING
# ==========================================
print("### Data Info Summary ###")
print(df.info())

print("\n### Checking for Missing Values ###")
print(f"Total missing values found: {df.isnull().sum().sum()}")

print("\n### Checking for Duplicate Rows ###")
print(f"Total duplicate rows found: {df.duplicated().sum()}")

# Drop columns that do not contribute to predictive patterns
columns_to_drop = ['PatientID', 'DoctorInCharge']
existing_drops = [col for col in columns_to_drop if col in df.columns]
if existing_drops:
    df = df.drop(columns=existing_drops)
    print(f"\nDropped unique/confidential identifiers: {existing_drops}")

print("\n### Target Class Distribution (%) ###")
print(df['Diagnosis'].value_counts(normalize=True) * 100)

# ==========================================
# 3. DATA VISUALIZATION
# ==========================================
print("\nGenerating Data Visualizations...")

# Plot 1: Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Diagnosis', palette='Set2')
plt.title('Distribution of Diabetes Diagnosis (0 = No, 1 = Yes)')
plt.xlabel('Diagnosis Status')
plt.ylabel('Count')
plt.show()

# Plot 2: Correlation Matrix for Key Clinical Indicators
clinical_cols = ['Age', 'BMI', 'FastingBloodSugar', 'HbA1c', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'Diagnosis']
plt.figure(figsize=(10, 8))
sns.heatmap(df[clinical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Key Clinical Measurements')
plt.show()

# Plot 3: HbA1c vs Fasting Blood Sugar by Diagnosis
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='FastingBloodSugar', y='HbA1c', hue='Diagnosis', alpha=0.7, palette='coolwarm')
plt.title('HbA1c vs Fasting Blood Sugar Distribution')
plt.xlabel('Fasting Blood Sugar (mg/dL)')
plt.ylabel('HbA1c (%)')
plt.show()

# ==========================================
# 4. MACHINE LEARNING PIPELINE
# ==========================================
print("\n--- Building Predictive Model ---")

# Separate features (X) and target variable (y)
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Split the dataset into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Standardize the features for uniform scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# 5. MODEL EVALUATION & INSIGHTS
# ==========================================
print("\n### Model Performance Evaluation ###")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visual
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'])
plt.title('Confusion Matrix')
plt.show()

# ROC-AUC Curve
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Feature Importance Extraction
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

# Plot Top 15 Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices[:15]], y=feature_names[indices[:15]], palette='viridis')
plt.title('Top 15 Most Important Features in Predicting Diabetes')
plt.xlabel('Relative Importance Score')
plt.ylabel('Features')
plt.show()
