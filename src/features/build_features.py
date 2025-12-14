import pandas as pd

def make_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store','item','date'])

    # calendar
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # lag
    for lag in [7,14,28]:
        df[f'lag_{lag}'] = df.groupby(['store','item'])['sales'].shift(lag)

    # rolling
    for w in [7,14,28]:
        df[f'rolling_mean_{w}'] = (
            df.groupby(['store','item'])['sales']
            .shift(1)
            .rolling(w)
            .mean()
        )

    return df
