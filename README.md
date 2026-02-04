# Claims Analytics and Cost Prediction (Insurance)

Machine Learning project to predict medical insurance charges based on demographic and behavioral data.

---

## Objective

The objective of this project is to develop a predictive model capable of estimating insurance medical costs using key variables such as:

- age  
- BMI  
- smoking status  
- region  
- number of children  

---

## Dataset

The dataset used in this project is **insurance.csv**, which contains the following variables:

- **age** – age of the primary beneficiary  
- **sex** – gender (male/female)  
- **bmi** – body mass index  
- **children** – number of dependents  
- **smoker** – smoking status  
- **region** – residential area  
- **charges** – medical insurance cost (target variable)

---

## Methodology

The project follows a structured machine learning workflow:

1. Data loading and exploratory analysis  
2. Preprocessing of numerical and categorical variables  
3. Splitting data into training and testing sets  
4. Building a regression pipeline using Random Forest  
5. Evaluating model performance  
6. Generating predictions and analyzing feature importance  

---

## Model Performance

The final model achieved the following evaluation metrics:

| Metric | Value |
|------|------|
| RMSE | 4603.87 |
| MAE  | 2525.65 |
| R²   | 0.863 |
| Training samples | 1070 |
| Test samples | 268 |

These results demonstrate strong predictive performance and confirm the model’s reliability for estimating insurance costs.

---

## Feature Importance

Analysis of feature importance revealed the following key drivers of medical charges:

| Feature | Importance |
|--------|-----------|
| smoker (no) | 0.425 |
| bmi | 0.210 |
| smoker (yes) | 0.184 |
| age | 0.134 |
| children | 0.019 |

Smoking status and BMI are clearly the most influential factors in determining insurance costs.

---

## Project Structure

insurance-claims-analytics/  
│  
├── data/  
│   └── insurance.csv  
│  
├── src/  
│   ├── train_model.py  
│   ├── evaluate.py  
│   ├── predict.py  
│   ├── utils.py  
│   └── config.py  
│  
├── outputs/  
│   ├── metrics.json  
│   ├── feature_importance.csv  
│   ├── predictions_sample.csv  
│   ├── prediction_one_example.csv  
│   └── model.joblib  
│  
├── notebooks/  
│   └── claims_eda.ipynb  
│  
├── requirements.txt  
└── README.md  

---

## How to Run

### 1) Install dependencies

Run the following command from the project root directory:

```bash
pip install -r requirements.txt
Tech Stack
Python

Pandas

NumPy

Scikit-learn

Joblib

Conclusions
A Random Forest regression model was successfully implemented to predict insurance medical charges.

The model achieved an R² score of 0.863, indicating strong predictive capability.

Smoking status, BMI, and age were identified as the most important predictors of medical costs.

The project provides an automated, modular, and reproducible machine learning pipeline for insurance cost estimation.

The results demonstrate how machine learning can support data-driven decision-making in the healthcare and insurance domains.

Author
Denisse Pareja
Data Scientist – TripleTen