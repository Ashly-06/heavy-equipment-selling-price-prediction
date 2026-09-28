# Heavy Equipment Selling Price Prediction

A machine learning regression project to predict heavy equipment selling prices using equipment specifications, usage information, transaction details, and categorical features.

## Project Overview

This project develops and evaluates tree-based regression models for predicting equipment selling prices from structured equipment and transaction data.

The workflow includes exploratory data analysis, missing-value analysis, outlier analysis, feature engineering, preprocessing, hyperparameter tuning, model comparison, and ensemble prediction.

## Dataset

- Training records: 138,701
- Test records: 15,000
- Target variable: `TargetValue`
- Evaluation metric: RMSLE (Root Mean Squared Logarithmic Error)
- Source: Kaggle Heavy Equipment Selling Price Prediction Challenge

## Exploratory Data Analysis

The analysis includes:

- Dataset structure and missing-value analysis
- Target value distribution
- Log-transformed target distribution
- Numerical feature correlation analysis
- Numerical feature outlier analysis
- Asset age distribution
- Asset age vs. target value relationship
- Model performance comparison
- Actual vs. predicted target values

The EDA visualizations and interpretations are included in the project notebook.

## Data Preprocessing

The following preprocessing steps were performed:

- Converted transaction dates to datetime format
- Replaced invalid manufacture-year values (`1000` and `1001`) with missing values
- Imputed missing numerical values using the median
- Imputed missing categorical values using `"Missing"`
- Encoded categorical variables using `OrdinalEncoder`
- Standardized numerical variables
- Applied log transformation using `log1p()` to the target variable

## Feature Engineering

The following features were engineered:

- `SaleYear`
- `SaleMonth`
- `SaleQuarter`
- `SaleDay`
- `SaleWeekday`
- `AssetAge`
- `HasOperationalHours`
- `HoursPerYear`
- `DescriptorLength`

These features capture transaction timing, equipment age, usage intensity, availability of operational-hour information, and descriptor characteristics.

## Machine Learning Models

Three tree-based regression models were trained and evaluated:

- CatBoost
- LightGBM
- XGBoost

CatBoost hyperparameters were tuned using `RandomizedSearchCV` with 3-fold cross-validation.

A weighted ensemble was also created using:

- CatBoost: 30%
- LightGBM: 50%
- XGBoost: 20%

## Model Performance

Models were evaluated using an 80/20 train-validation split and RMSLE.

| Model | Validation RMSLE |
|---|---:|
| CatBoost | 0.2367 |
| LightGBM | 0.2189 |
| XGBoost | 0.2193 |
| Weighted Ensemble | 0.2204 |

LightGBM achieved the lowest validation RMSLE among the individual models.

The final submission was generated using the weighted ensemble of CatBoost, LightGBM, and XGBoost.

## Kaggle Result

**Kaggle Leaderboard RMSLE: 0.1918**

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- LightGBM
- XGBoost
- Matplotlib
- Seaborn

## Repository Contents

```text
heavy-equipment-selling-price-prediction/
│
├── README.md
├── Heavy_Equipment_Selling_Price_Prediction.ipynb
└── requirements.txt
```

## Notebook

[View Complete Notebook](./heavy-equipment-selling-price-prediction_py.ipynb)
