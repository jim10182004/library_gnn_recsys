# Hugging Face Spaces 部署指南

把這個專案部署到 [Hugging Face Spaces](https://huggingface.co/spaces) 後，就會有一個公開網址（如 `https://huggingface.co/spaces/your-name/library-gnn`）任何人都能用。

**免費方案**：CPU 16GB RAM，大小限制 50GB，不錯用。

## 步驟

### 1. 註冊 Hugging Face 帳號
[https://huggingface.co/join](https://huggingface.co/join)

### 2. 建立新 Space
1. 點 [Create new Space](https://huggingface.co/new-space)
2. **Space name**: `library-gnn-recsys`（或自訂）
3. **License**: MIT
4. **Space SDK**: **Streamlit**
5. **Space hardware**: CPU basic（免費）
6. **Visibility**: Public
7. 點 **Create Space**

### 3. 把專案推上去

HF Space 本質就是個 git repo。

```powershell
# 在專案根目錄
git init
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/library-gnn-recsys

# 建立 README.md（HF Spaces 需要 metadata header）
# 把下方的 README_FOR_HF.md 改名為 README.md（或合併內容）

# 加入要部署的檔案（不要全部，只要 demo 需要的）
git add app_public.py requirements.txt README.md
git add src/dataset.py src/evaluate.py
git add src/models/lightgcn.py src/models/__init__.py src/__init__.py
git add data/processed/books.parquet
git add data/splits/train.parquet data/splits/val.parquet data/splits/test.parquet
git add data/splits/meta.json data/splits/user_remap.parquet data/splits/item_remap.parquet
git add checkpoints/lightgcn_best.pt

git commit -m "Initial deploy"
git push hf main
```

### 4. 等部署完成
HF Spaces 會自動讀 `requirements.txt`、安裝依賴、執行 `streamlit run app_public.py`。

第一次部署約 5-10 分鐘。完成後點 **App** 標籤即可看 demo。

---

## 注意事項

### 隱私
- Books.parquet 裡有完整書名/作者，這 OK，因為書本資訊不是個人資料。
- **不要**上傳 borrows.parquet、reservations.parquet（含借閱日期等）
- 已透過 `app_public.py` 的設計避免顯示真實讀者 ID

### 大小檢查
建議部署版只包含必要檔案：
```
books.parquet         ~9 MB
splits/*              ~5 MB
lightgcn_best.pt     ~10 MB
程式碼                ~50 KB
總計：               ~25 MB（遠低於 50GB 限制）
```

### 不要部署 BERT 版本
`book_bert.parquet` 234 MB 會讓 build 變慢。如果要上 BERT 版，改用 GPU Spaces（付費）。

---

## README.md（給 HF 用的範本）

```markdown
---
title: Library GNN Recommendation
emoji: 📚
colorFrom: teal
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app_public.py
pinned: false
license: mit
---

# 圖書館 GNN 推薦系統

基於 LightGCN（SIGIR 2020）的個人化書籍推薦系統，畢業專題作品。

## 功能
- 5 種預設讀者原型（Persona）的即時推薦展示
- 自訂喜歡的書 → 推薦相似書籍
- 模型在某市立圖書館（已去識別化）2025 年資料上訓練

## 技術
- PyTorch 2.6 + LightGCN 3 層
- 35,856 讀者 × 29,685 書 × 525,288 互動

## 隱私聲明
所有讀者資料已匿名化，本 demo 不顯示任何真實讀者 ID。
```

把上面內容存為 `README.md` 放到專案根目錄。
