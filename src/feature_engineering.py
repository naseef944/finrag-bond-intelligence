import pandas as pd

def create_features(df):
    # example: create month, quarter, rolling stats
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['issuance_roll_mean_2'] = df['issuance_volume'].rolling(window=2, min_periods=1).mean().values
    return df

