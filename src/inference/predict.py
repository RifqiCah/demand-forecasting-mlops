import json
import pandas as pd
import lightgbm as lgb
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# src/inference/predict.py
# parent.parent.parent = project root

MODEL_PATH = BASE_DIR / "models" / "lgb_demand_model.txt"
FEATURE_PATH = BASE_DIR / "models" / "features.json"

model = lgb.Booster(model_file=str(MODEL_PATH))

with open(FEATURE_PATH) as f:
    FEATURES = json.load(f)



def predict_next_days(history_df, n_days=7):
    """
    history_df: dataframe with past data (already feature-engineered)
    """
    last_date = history_df['date'].max()
    preds = []

    df = history_df.copy()

    for i in range(n_days):
        next_date = last_date + timedelta(days=1)

        row = df.iloc[-1:].copy()
        row['date'] = next_date

        # update calendar features
        row['year'] = next_date.year
        row['month'] = next_date.month
        row['week'] = next_date.isocalendar()[1]
        row['day'] = next_date.day
        row['dayofweek'] = next_date.weekday()
        row['is_weekend'] = int(row['dayofweek'].iloc[0] in [5, 6])

            # 🔥 LAG FEATURES (INI KUNCI)
        row['lag_7'] = df['sales'].iloc[-7]
        row['lag_14'] = df['sales'].iloc[-14]

        # 🔥 ROLLING FEATURES
        row['rolling_mean_7'] = df['sales'].iloc[-7:].mean()
        row['rolling_mean_14'] = df['sales'].iloc[-14:].mean()

        X = row[FEATURES]
        pred = model.predict(X)[0]

        row['sales'] = pred
        df = pd.concat([df, row], ignore_index=True)

        preds.append((next_date, pred))

        last_date = next_date


    return preds
