from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBED_DIR = DATA_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "models"
TAB_MODEL_PATH = MODELS_DIR / "tabular" / "forecast_model.pkl"
VECTORSTORE_DIR = MODELS_DIR / "vector_store"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VERSION=1

# Embedding & retrieval
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Local lightweight LLM by default — replace with local path to larger model if available
LLM_MODEL = os.getenv("LLM_MODEL", "distilgpt2")

# RAG settings
CHUNK_SIZE = 400
TOP_K = 3

