import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from .data_loader import merge_and_prepare
from .config import TAB_MODEL_PATH

def shap_explain():
    df = merge_and_prepare()
    df = df.dropna()
    X = df[['issuance_volume', 'spread', 'gdp_growth', 'inflation', 'unemployment']]
    model = joblib.load(TAB_MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # save a summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("docs/shap_summary.png")
    print("Saved SHAP summary to docs/shap_summary.png")

def dowhy_stub():
    # full causal pipeline requires careful identification; here is a placeholder stub
    print("DoWhy causal analysis should be performed in notebooks with target causal graph specification.")

