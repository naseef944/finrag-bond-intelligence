import pandas as pd
from .config import RAW_DIR, PROCESSED_DIR
from pathlib import Path

def load_bond_data():
    fn = Path(RAW_DIR) / "bond_issuance.csv"
    df = pd.read_csv(fn, parse_dates=["date"])
    return df

def load_macro_data():
    fn = Path(RAW_DIR) / "macro_indicators.csv"
    df = pd.read_csv(fn, parse_dates=["date"])
    return df

def merge_and_prepare():
    bonds = load_bond_data()
    macro = load_macro_data()
    # aggregate macro to quarterly by mean
    macro_q = macro.set_index("date").resample("Q").mean().reset_index()
    merged = pd.merge(bonds, macro_q, how="left", on="date")
    # create a lagged target example
    merged["target_next_issuance"] = merged["issuance_volume"].shift(-1)
    processed_fn = Path(PROCESSED_DIR)/"processed_dataset.csv"
    merged.dropna().to_csv(processed_fn, index=False)
    print(f"Saved processed dataset to {processed_fn}")
    return merged

