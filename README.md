# Customer Churn Prediction Using Machine Learning

This project develops an end-to-end machine learning pipeline to predict customer churn using the IBM Telco Customer Churn dataset.

## Project Aim

To identify customers likely to churn using demographic, service, contract, and billing features, and evaluate machine learning models using professional classification metrics.

## Dataset

IBM Telco Customer Churn dataset.

Target variable: `Churn`

## Repository Structure

- `data/raw/` original dataset
- `data/processed/` cleaned dataset
- `notebooks/` exploratory analysis
- `src/` reusable ML pipeline scripts
- `outputs/figures/` visual outputs
- `outputs/tables/` model result tables
- `outputs/models/` trained models
- `report/` final assessment report

## Models

- Logistic Regression
- Random Forest
- XGBoost

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix