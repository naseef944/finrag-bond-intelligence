import argparse
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import random
from src.config import RAW_DIR

def generate_bond_issuance(n_years=5):
    dates = pd.date_range(end=datetime.today(), periods=n_years*4, freq='Q')
    data = []
    for d in dates:
        # simulate issuance volume and spread
        base = 200 + (d.year - dates[0].year) * 2
        issuance = round(base + random.uniform(-30, 30), 1)
        spread = round(0.02 + random.uniform(-0.01, 0.03), 4)
        data.append({"date": d.strftime("%Y-%m-%d"), "issuance_volume": issuance, "spread": spread})
    df = pd.DataFrame(data)
    fn = RAW_DIR / "bond_issuance.csv"
    df.to_csv(fn, index=False)
    print(f"Generated {fn}")

def generate_macro(n_years=5):
    dates = pd.date_range(end=datetime.today(), periods=n_years*12, freq='M')
    data = []
    for d in dates:
        gdp = round(2.0 + random.uniform(-1, 1), 2)
        inflation = round(1.5 + random.uniform(-1.0, 2.0), 2)
        unemployment = round(5.0 + random.uniform(-1.5, 1.5), 2)
        data.append({"date": d.strftime("%Y-%m-%d"), "gdp_growth": gdp, "inflation": inflation, "unemployment": unemployment})
    df = pd.DataFrame(data)
    fn = RAW_DIR / "macro_indicators.csv"
    df.to_csv(fn, index=False)
    print(f"Generated {fn}")

def generate_news():
    articles = [
        {"id":"n1", "date":"2025-10-01", "source":"ECB", "title":"Inflation rises across euro area", "text":"Inflation increased this month... central bank watches yields."},
        {"id":"n2", "date":"2025-10-05", "source":"Reuters", "title":"Sovereign issuance stable", "text":"Several countries issued bonds with moderate demand..."},
        {"id":"n3", "date":"2025-10-10", "source":"Financial Times", "title":"Investors wary of spreads", "text":"Spreads slightly widened after economic data..."}
    ]
    df = pd.DataFrame(articles)
    fn = RAW_DIR / "news_articles.csv"
    df.to_csv(fn, index=False)
    print(f"Generated {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate_bond_issuance()
        generate_macro()
        generate_news()

