# 多階段 Dockerfile：可同時用於 Streamlit / FastAPI 部署
#
# 建置：
#     docker build -t library-gnn .
# 執行 Streamlit：
#     docker run -p 8501:8501 library-gnn
# 執行 FastAPI：
#     docker run -p 8000:8000 library-gnn uvicorn api.main:app --host 0.0.0.0 --port 8000

FROM python:3.10-slim

WORKDIR /app

# 系統套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴（先 cache 這層）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit fastapi uvicorn

# 複製專案檔
COPY src/ ./src/
COPY app_public.py ./
COPY api/ ./api/
# data 與 checkpoints 用 volume 或 build 時掛入
COPY data/processed/ ./data/processed/
COPY data/splits/ ./data/splits/
COPY checkpoints/lightgcn_best.pt ./checkpoints/

# 預設執行 Streamlit
EXPOSE 8501 8000
CMD ["streamlit", "run", "app_public.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
