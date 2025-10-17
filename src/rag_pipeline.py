import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .config import EMBED_MODEL, VECTORSTORE_DIR, EMBED_DIR, LLM_MODEL, TOP_K

VECTORSTORE_DIR = Path(VECTORSTORE_DIR)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

def build_index(news_csv=None):
    if news_csv is None:
        news_csv = Path("data/raw/news_articles.csv")
    df = pd.read_csv(news_csv)
    texts = df['text'].astype(str).tolist()
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(VECTORSTORE_DIR / "faiss.index"))
    np.save(VECTORSTORE_DIR / "texts.npy", np.array(texts, dtype=object))
    df.to_csv(VECTORSTORE_DIR / "meta.csv", index=False)
    print("Built FAISS index at", VECTORSTORE_DIR)

def load_index():
    index = faiss.read_index(str(VECTORSTORE_DIR / "faiss.index"))
    texts = np.load(VECTORSTORE_DIR / "texts.npy", allow_pickle=True)
    meta = pd.read_csv(VECTORSTORE_DIR / "meta.csv")
    return index, texts, meta

def generate_summary(query, max_len=200):
    index, texts, meta = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL)
    q_emb = embed_model.encode([query]).astype('float32')
    D, I = index.search(q_emb, TOP_K)
    ctx = "\n\n".join([texts[int(i)] for i in I[0]])
    # local lightweight LLM pipeline
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    prompt = f"Context:\n{ctx}\n\nSummarize the main points in concise bullet points:"
    out = gen(prompt, max_length=max_len, do_sample=False)
    return {"query": query, "summary": out[0]["generated_text"], "sources": I[0].tolist()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()
    if args.build_index:
        build_index()
    if args.query:
        res = generate_summary(args.query)
        print(res["summary"])

