FROM python:3.10-slim

WORKDIR /app

# 系統相依
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# 先複製 requirements 利用快取
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式 + assets + 前端
COPY app.py .
COPY assets/ ./assets/
COPY static/ ./static/

# HF Spaces 預設 port
EXPOSE 7860

# 用 uvicorn 跑（FastAPI 入口在 app.py 的 app 物件）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
