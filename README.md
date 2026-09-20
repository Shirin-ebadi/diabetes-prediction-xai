# Diabetes Prediction with Machine Learning and Explainable AI

A multiclass machine learning project for predicting diabetes status using the BRFSS2015 health indicators dataset.

The project compares multiple machine learning algorithms and an Artificial Neural Network (ANN), handles class imbalance using SMOTE, and uses SHAP for model explainability.

## Problem

The goal is to classify individuals into three categories:

- Healthy
- Prediabetes
- Diabetes

The target variable is `Diabetes_012`.

## Dataset

The project uses the **BRFSS2015 Diabetes Health Indicators Dataset**.

The dataset includes health, behavioral, and demographic features such as:

- BMI
- High blood pressure
- High cholesterol
- Physical activity
- General health
- Age
- Smoking status
- Fruit and vegetable consumption

The dataset is downloaded automatically through KaggleHub in the code.

## Preprocessing

The pipeline includes:

- Random sampling of 15,000 records
- Stratified train/test split
- 70% training and 30% testing
- StandardScaler normalization
- SMOTE applied only to the training data
- Prevention of data leakage

## Models

The following models are evaluated:

- Artificial Neural Network (ANN)
- Random Forest
- Gradient Boosting
- Decision Tree
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
- AdaBoost

## Evaluation Metrics

Models are compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- Mean Squared Error (MSE)

## Explainable AI

SHAP is used to explain the Random Forest model.

The analysis includes:

- Global feature importance
- SHAP bar plot
- SHAP beeswarm plot
- Analysis of features influencing the diabetes class

Important features identified in the project include:

- `GenHlth`
- `HighBP`
- `HighChol`

## Results

In the project report, Gradient Boosting achieved the highest reported accuracy:

| Model | Accuracy |
|---|---:|
| Gradient Boosting | 79.71% |
| Random Forest | 79.06% |
| Decision Tree | 76.22% |
| AdaBoost | 70.02% |
| SVM | 64.33% |
| ANN | 64.28% |
| KNN | 63.88% |
| Logistic Regression | 61.22% |
| Naive Bayes | 53.66% |

## Files

- `diabetes_prediction_xai.py` — Complete ML, ANN, preprocessing, evaluation and SHAP pipeline
- `README.md` — Project documentation

## Requirements

Main dependencies:

```bash
pip install pandas numpy matplotlib seaborn shap scikit-learn imbalanced-learn tensorflow kagglehub
