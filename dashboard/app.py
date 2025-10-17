import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.title("FinRAG — Bond Market Intelligence (Demo)")

st.markdown("Enter a query to get an automated RAG-based summary and a sample forecast.")

query = st.text_input("Query", "sovereign bond issuance outlook euro area")

if st.button("Generate"):
    with st.spinner("Generating..."):
        res = requests.post("http://localhost:8000/generate", json={"query": query}).json()
    st.subheader("RAG Summary")
    st.write(res.get("rag_summary"))
    st.subheader("Forecast")
    st.json(res.get("forecast"))
    st.subheader("Top drivers")
    st.json(res.get("top_drivers"))

# Example forecast plot if processed data exists
try:
    df = pd.read_csv("data/processed/processed_dataset.csv", parse_dates=["date"])
    fig = px.line(df, x="date", y="issuance_volume", title="Issuance Volume (historical)")
    st.plotly_chart(fig)
except Exception:
    st.info("No processed dataset found. Run sample data generator.")

