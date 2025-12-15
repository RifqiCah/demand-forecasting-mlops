# Store Item Demand Forecasting (MLOps Project)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-CatBoost%20|%20LightGBM%20|%20XGBoost-orange)
![Metric](https://img.shields.io/badge/Best%20SMAPE-11.95-success)

## 📌 Project Overview
This project aims to predict future sales demand for 50 different items across 10 different stores for a 3-month period. The dataset is sourced from the Kaggle **Store Item Demand Forecasting Challenge**.

This solution implements a structured **MLOps** workflow, leveraging advanced Time Series Feature Engineering (Lags & Rolling Windows) and an **Ensemble Learning** strategy using modern Gradient Boosting Decision Trees (GBDT).

## 📂 Project Structure
The project follows a standard MLOps/Cookiecutter Data Science directory structure:

```text
DEMAND-FORECASTING-MLOPS/
├── data/
│   ├── raw/               # Original datasets (train.csv, test.csv)
│   └── processed/         # Feature-engineered data (train_fe.csv, test_fe.csv)
├── experiments/           # Model diagnostics (Learning Curves, Feature Importance plots)
├── models/                # Saved model artifacts (.cbm, .pkl, .json)
├── notebooks/             # Jupyter Notebooks for experimentation
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_ensemble.ipynb
├── src/                   # Source code (optional scripts)
├── submissions/           # Generated submission CSVs for Kaggle
└── README.md

🧠 Methodology
1. Feature EngineeringFeatures were engineered to capture seasonal patterns and short-term trends:
- Time Features: dayofweek, month, year, is_weekend.
- Lag Features: lag_7, lag_14, lag_28 (Sales from the same day last week/month).
- Rolling Mean Features: rolling_mean_7, rolling_mean_14, rolling_mean_28 (Moving average trends).
2. Models UsedWe experimented with three state-of-the-art Gradient Boosting models:
- CatBoost (Champion): Achieved the best single-model performance due to its superior handling of categorical features (store, item).
- LightGBM: Used as a fast and efficient baseline.
- XGBoost: Used to provide diversity for the ensemble.
3. Ensemble StrategyThe final submission utilizes a Weighted Blending strategy to maximize accuracy and generalization:$$ FinalPrediction = (0.8 \times CatBoost) + (0.2 \times LightGBM)
Model Strategy,Validation SMAPE,Notes
CatBoost (Single),11.9539,Best Single Model
Ensemble (Cat+LGB+XGB),11.9782,Slightly higher due to noise from weaker models
Ensemble (80% Cat + 20% LGB),11.9537,Best Score (Selected for Submission)
LightGBM (Single),12.0366,Strong baseline
XGBoost (Single),12.0542,-

Key Insight: Feature importance analysis reveals that rolling_mean_7 and rolling_mean_14 are the most critical predictors, indicating the model relies heavily on recent short-term trends.
Model DiagnosticsVisual diagnostic artifacts are saved in the experiments/ directory: 1_learning_curves_comparison.png: Monitors overfitting/underfitting behavior. 2_total_daily_sales_comparison.png: Visualizes aggregated predicted vs. actual trends. 3_catboost_feature_importance.png: Top influential features (Gain/Split).
How to RunClone
Repository:Bashgit clone [https://github.com/your-username/repo-name.git](https://github.com/your-username/repo-name.git)
cd repo-name
Install Dependencies:Bashpip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost joblib
Run Notebooks:Execute the notebooks in the notebooks/ directory in the following order:
Run 02_feature_engineering.ipynb to generate processed data in data/processed/.
Run 03_modeling_ensemble.ipynb to train models, evaluate performance, and generate submission files.
Author[Rifqi Cahyono - AI/ML Engineer
