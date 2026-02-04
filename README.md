# Claims Analytics and Cost Prediction

Machine learning project to predict medical insurance costs using demographic and health-related variables.

---

## Objective

Build a regression model capable of estimating insurance charges based on client characteristics such as:

- Age  
- BMI  
- Smoking status  
- Number of children  
- Region  
- Sex  

The goal is to support insurance companies in:

- Pricing policies more accurately  
- Detecting high-risk profiles  
- Improving financial planning  

---

## Dataset

The project uses the classic insurance dataset containing:

- 1,338 records  
- 7 features  
- 1 target variable: `charges`  

Data is split into:

- Train: 1070 samples  
- Test: 268 samples  

---

## Approach

1. Data loading and preprocessing  
2. Categorical encoding with OneHotEncoder  
3. Numeric scaling  
4. Pipeline-based modeling  
5. Random Forest Regressor  
6. Model evaluation  
7. Feature importance analysis  
8. Prediction generation  

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  

---

## Results

### Model Performance

| Metric | Value |
|------|------|
| RMSE | 4603.87 |
| MAE | 2525.65 |
| R² Score | 0.863 |
| Train Samples | 1070 |
| Test Samples | 268 |

**Interpretation**

- The model explains **86.3% of the variance** in insurance costs.  
- On average, predictions deviate around **$2,525 USD** from real charges.  
- Performance is strong for a baseline structured ML approach.

---

## Feature Importance

Top factors influencing predicted insurance cost:

| Feature | Importance |
|--------|-----------|
| smoker (no) | 0.425 |
| BMI | 0.210 |
| smoker (yes) | 0.185 |
| age | 0.134 |
| children | 0.019 |
| region | < 0.01 |
| sex | < 0.01 |

**Key Insight:**  
Smoking status and BMI are by far the strongest drivers of insurance charges.

---

## Outputs Generated

After training, the system automatically creates:

- `outputs/model.joblib` – trained model  
- `outputs/metrics.json` – evaluation metrics  
- `outputs/predictions_sample.csv` – batch predictions  
- `outputs/prediction_one_example.csv` – single prediction example  
- `outputs/feature_importance.csv` – importance analysis  

---

## How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
