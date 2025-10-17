from fastapi import FastAPI
from pydantic import BaseModel
from ..tabular_models import train_model
from ..rag_pipeline import generate_summary
from ..report_generator import build_report
import joblib
import pandas as pd
from ..config import TAB_MODEL_PATH

app = FastAPI(title="FinRAG API")

class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/generate")
def generate(q: Query):
    # quick RAG summary
    rag = generate_summary(q.query)
    # quick example forecast (load model & do a dummy forecast)
    # In a full pipeline you'd pass scenario features; here return placeholder
    try:
        model = joblib.load(TAB_MODEL_PATH)
        # for demo, predict last row
        df = pd.read_csv("data/processed/processed_dataset.csv")
        x = df[['issuance_volume','spread','gdp_growth','inflation','unemployment']].iloc[-1:].fillna(0)
        pred = model.predict(x)[0].item()
        forecast = {"next_issuance": float(pred)}
    except Exception as e:
        forecast = {"error": str(e)}

    report = build_report(forecast, rag['summary'])
    return report

