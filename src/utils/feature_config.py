BASE_FEATURES = [
    'store','item','year','month','week','day',
    'dayofweek','is_weekend'
]

LAG_FEATURES = ['lag_7','lag_14','lag_28']
ROLLING_FEATURES = ['rolling_mean_7','rolling_mean_14','rolling_mean_28']

FEATURES = BASE_FEATURES + LAG_FEATURES + ROLLING_FEATURES
TARGET = 'sales'
