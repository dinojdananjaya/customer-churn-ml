## Customer Churn Prediction Using Machine Learning

This repository contains an end-to-end machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset. The project was developed for the CSC-44112 technical data science assessment and is structured to support reproducible analysis, professional model development, and clear technical reporting.

## Project Aim

The aim of this project is to develop and evaluate a supervised machine learning pipeline that can identify customers who are likely to churn. The analysis focuses not only on predictive performance, but also on interpretability, responsible use, and real-world business value.

## Dataset

The dataset used is the IBM Telco Customer Churn dataset.

- Source: IBM sample dataset
- Target variable: `Churn`
- Task type: Binary classification
- Main feature groups: customer demographics, account information, service usage, contract type, and billing details

## Repository Structure

```text
customer-churn-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── tables/
│
├── report/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── utils.py
│   └── visualisation.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

## Methodology Overview

# This project follows a structured machine learning workflow:

Raw dataset ingestion
Data cleaning and validation
Exploratory Data Analysis (EDA)
Train-test dataset splitting
Feature preprocessing and transformation
Machine learning model training
Hyperparameter optimisation
Model evaluation and comparison
Explainable AI analysis using SHAP

## Models Used

# The following classification models are trained and compared:

Logistic Regression
Random Forest Classifier
XGBoost Classifier

The final model is selected using cross-validated ROC-AUC and then evaluated on an unseen test set.

## Evaluation Metrics

# The project evaluates classification performance using:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion matrix
ROC curve
Precision-recall curve

These metrics were selected because churn prediction involves a business trade-off between identifying customers at risk of leaving and avoiding unnecessary retention interventions.

## How to Run the Project

Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the full pipeline:

python main.py

Alternatively, run individual stages:

python src/data_loader.py
python src/eda.py
python src/train.py
python src/evaluate.py
python src/explainability.py

## Main Outputs

# Generated outputs are saved in the outputs/ directory:

outputs/tables/ contains EDA summary tables
outputs/figures/ contains report-ready visualisations
outputs/metrics/ contains model comparison and evaluation results
outputs/models/ contains trained model files

## Responsible AI and Professional Considerations

This project uses a public fictional customer dataset and does not contain personally identifiable real customer data. The report discusses limitations around data representativeness, possible bias, model reliability, explainability, and practical deployment considerations.