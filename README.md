# FinRAG — AI-Assisted Bond Market Intelligence

Prototype that combines tabular ML forecasting, causal inference, and RAG-enabled LLM summarization for sovereign bond market intelligence.

## Quickstart (local)
1. Create venv & install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Generate sample data (or fetch ECB):
   python src/sample_data.py --generate

3. Build embeddings:
   python -m src.rag_pipeline --build-index

4. Train tabular model:
   python -m src.tabular_models --train

5. Run API:
   uvicorn src.api.main:app --reload --port 8000

6. Run Dashboard:
   streamlit run dashboard/app.py

## LLM (local)
Default LLM: `distilgpt2` (lightweight). To use another local model, set LLM_MODEL in `src/config.py` or export env var `LLM_MODEL`.

## License
MIT
