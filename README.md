# Healthcare Claims Analytics & Cost Prediction Framework

## Executive Summary

This project analyzes healthcare insurance claims data to identify key cost drivers and build a predictive framework that supports risk segmentation, financial forecasting, and preventive strategy design.

The goal is not only to predict costs, but to translate analytical findings into strategic decision-making insights.

---

## Business Problem

Healthcare insurers and provider organizations must understand which variables most strongly influence claim costs in order to:

Improve financial planning

Reduce unexpected cost variability

Design targeted preventive interventions

Allocate resources efficiently

Without structured analytics, cost management becomes reactive rather than strategic.

---

## Key Insights

Smoking status is one of the strongest cost predictors

Higher BMI significantly increases medical expense variability

Age contributes to compounding cost risk

Risk factors intensify when combined

---

## Strategic Recommendations

Segment insured populations by risk exposure

Prioritize preventive programs for high-risk groups

Implement early cost monitoring for compounding risk profiles

Use predictive modeling to inform premium strategy and budgeting

---

## Business Impact

This analytical framework enables organizations to:

Improve cost prediction accuracy

Identify high-risk populations earlier

Support data-driven financial planning

Strengthen risk management strategy

---

## Stakeholder Value

This project demonstrates how different organizational stakeholders can leverage predictive analytics:

- Finance teams → forecast costs and reduce financial uncertainty  
- Clinical teams → identify high-risk populations early  
- Strategy teams → design prevention-focused initiatives  
- Executives → make data-informed policy and pricing decisions  

---

## Project Description

Machine Learning project to predict medical insurance charges based on demographic and behavioral data.

---

## Objective

The objective of this project is to develop a predictive model capable of estimating insurance medical costs using key variables such as age, BMI, smoking status, region, and number of children.

---

## Dataset

The dataset used in this project is `insurance.csv`, which contains the following variables:

- age – age of the primary beneficiary  
- sex – gender (male/female)  
- bmi – body mass index  
- children – number of dependents  
- smoker – smoking status  
- region – residential area  
- charges – medical insurance cost (target variable)

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

## Interpretation of Results

The model demonstrates strong predictive performance, indicating that healthcare cost variability can be reliably estimated using a combination of demographic and behavioral variables.

The results suggest that cost prediction in healthcare is not random, but largely driven by identifiable risk factors that organizations can monitor and manage proactively.

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

## Analytical Interpretation

Feature importance analysis confirms that behavioral and physiological variables play a significantly larger role in cost prediction than demographic variables alone.

This suggests that risk management strategies should focus on modifiable health factors rather than static characteristics.

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
```

### 2) Train the model

```bash
python src/train_model.py
```

This will generate the following outputs:

- outputs/model.joblib  
- outputs/metrics.json  
- outputs/predictions_sample.csv  
- outputs/feature_importance.csv  

### 3) Make predictions

```bash
python src/predict.py
```

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  

---

## Limitations & Assumptions

This model is based on a structured dataset with limited behavioral variables and does not incorporate clinical history, geographic socioeconomic factors, or longitudinal health data.

Therefore, predictions should be interpreted as directional estimates rather than absolute forecasts.

Future iterations could improve performance by incorporating real-world healthcare datasets with longitudinal patient information.

---

## Conclusions

- A Random Forest regression model was successfully implemented to predict insurance medical charges.  
- The model achieved an **R² score of 0.863**, indicating strong predictive capability.  
- Smoking status, BMI, and age were identified as the most important predictors of medical costs.  
- The project provides an automated, modular, and reproducible machine learning pipeline for insurance cost estimation.  
- The results demonstrate how machine learning can support data-driven decision-making in the healthcare and insurance domains.

---

## Strategic Conclusions

This project demonstrates how predictive analytics can be used not only to estimate healthcare costs but also to support strategic decision-making.

The findings highlight that healthcare expenses are strongly influenced by identifiable risk factors such as smoking status, BMI, and age, which means organizations can move from reactive cost management to proactive risk monitoring.

By implementing structured analytics frameworks like this one, insurers and healthcare organizations can improve financial planning, optimize resource allocation, and strengthen preventive strategies.

---

## Executive Takeaway

Predictive healthcare cost modeling is most valuable when used as a decision-support system rather than a forecasting tool alone.

---

## Consulting Relevance

This framework can be adapted for insurers, healthcare providers, and health-tech organizations seeking structured analytics systems for risk evaluation, financial planning, and strategic decision-making.

---

## Author
**Denisse Pareja**  
Healthcare Analytics Consultant | Data Scientist  

Specialized in building decision-driven analytics systems that translate data into strategic insight for healthcare and operational organizations.

LinkedIn: https://linkedin.com/in/TUURL  
GitHub: https://github.com/denpareja

---

## Portfolio Note

This project is part of a professional analytics portfolio focused on developing decision-intelligence frameworks that help organizations move from descriptive reporting to predictive and strategic analytics.
