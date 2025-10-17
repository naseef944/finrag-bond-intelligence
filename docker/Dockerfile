FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential git
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000 8501
CMD ["bash","-lc","uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]

