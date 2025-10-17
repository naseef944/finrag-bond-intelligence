import argparse
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from .data_loader import merge_and_prepare
from .config import MODELS_DIR, TAB_MODEL_PATH

def train_model():
    df = merge_and_prepare()
    df = df.dropna()
    # simple features
    X = df[['issuance_volume', 'spread', 'gdp_growth', 'inflation', 'unemployment']].copy()
    y = df['target_next_issuance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print("RMSE:", rmse)
    Path(TAB_MODEL_PATH.parent).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TAB_MODEL_PATH)
    print("Saved model to", TAB_MODEL_PATH)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train_model()

