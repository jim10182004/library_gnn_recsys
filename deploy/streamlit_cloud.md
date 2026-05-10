# Streamlit Community Cloud 部署指南

[Streamlit Cloud](https://streamlit.io/cloud) 提供免費部署 Streamlit app 的服務，每個帳號可有多個 app。

**限制**：1 GB 記憶體、CPU only、檔案需在 GitHub。

## 步驟

### 1. 把專案推到 GitHub
（如果還沒）

```powershell
# 專案根目錄
git init
git add app_public.py requirements.txt
git add src/dataset.py src/evaluate.py src/__init__.py
git add src/models/lightgcn.py src/models/__init__.py
git add data/processed/books.parquet
git add data/splits/*
git add checkpoints/lightgcn_best.pt
git add .gitignore README.md

git commit -m "Initial commit"

# 在 GitHub 建立 repo: https://github.com/new
git remote add origin https://github.com/YOUR_USERNAME/library-gnn-recsys.git
git branch -M main
git push -u origin main
```

### 2. 在 Streamlit Cloud 建立 app
1. 到 [share.streamlit.io](https://share.streamlit.io)
2. 用 GitHub 登入
3. 點 **New app**
4. **Repository**: `YOUR_USERNAME/library-gnn-recsys`
5. **Branch**: `main`
6. **Main file path**: `app_public.py`
7. 點 **Deploy**

### 3. 等部署
約 3-5 分鐘。完成後給你一個 `https://your-app.streamlit.app` 網址。

---

## 注意事項

### Resource 限制
免費版限制：
- **Memory**: 1 GB
- **CPU**: 1 vCPU
- **Storage**: 1 GB
- **Bandwidth**: 沒限制但合理使用

LightGCN 載入後約使用 300-500 MB RAM，應該沒問題。
但若要部署多個模型（BERT、Hetero）會超出，請改用 HF Spaces。

### 大檔案處理
GitHub 單檔限制 100 MB。我們的檔案都小於這個。
若要上 `book_bert.parquet` (234 MB)，需用 [Git LFS](https://git-lfs.com/) 或改用 HF Spaces。

### Secrets（如果要保護）
如果要加密碼保護（避免被濫用）：
1. 在 Streamlit Cloud 控制台 → Secrets
2. 加入 `[passwords]` `student = "1234"`
3. 在 `app_public.py` 開頭加密碼檢查

---

## 常見問題

### Q: 部署後第一次很慢
A: 第一次 cold start 要載入模型（約 30 秒）。後續會 cache。

### Q: 部署失敗，requirements.txt 安裝錯誤
A: 確認 `requirements.txt` 沒指定 GPU 版的 torch。Streamlit Cloud 只有 CPU。
應該用：
```
torch --index-url https://download.pytorch.org/whl/cpu
```
或者直接 `pip install torch`（會自動裝 CPU 版）。

### Q: app 顯示 404
A: 部署中。等 5 分鐘後重整。
