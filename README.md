# 圖書館借閱資料 GNN 推薦系統 (Graduation Project)

## 專題目標
使用 **LightGCN**（Light Graph Convolutional Network）在某市立圖書館 2025 年借閱資料上做書籍推薦系統，並與傳統方法（Popular / ItemCF / BPR-MF）比較。

## 資料
- 借閱：約 130 萬筆（2025 全年）
- 預約：約 32 萬筆
- 讀者：10.9 萬位（已匿名化）
- 原始檔案位置：透過環境變數 `LIBRARY_RAW_DIR` 設定（見下方）

## 初次設定
1. 建議使用 Python 3.10 建立虛擬環境。
2. 安裝依賴：`pip install -r requirements.txt`
3. 複製 `.env.example` 為 `.env`
4. 編輯 `.env`，把 `LIBRARY_RAW_DIR` 設成你本機原始 Excel 資料夾路徑
5. 執行 `python src/preprocess.py` 將 Excel 轉為 Parquet
6. 執行 `python -m src.dataset` 切分 Train/Val/Test

> 若要使用 NVIDIA GPU，請先依照本機 CUDA 版本安裝對應的 PyTorch wheel，再安裝其餘套件。

## 資料夾結構
```
library_gnn_recsys/
├── data/
│   ├── raw/           # 原始資料 (參照桌面檔案，不複製)
│   ├── processed/     # 預處理後的 parquet
│   └── splits/        # train/val/test 切分
├── src/
│   ├── preprocess.py  # 資料前處理
│   ├── dataset.py     # PyG Dataset
│   ├── models/        # LightGCN + baselines
│   ├── train.py       # 訓練
│   └── evaluate.py    # 評估
├── notebooks/         # EDA / 視覺化 / Demo
├── checkpoints/       # 模型權重
└── results/           # 實驗結果與圖
```

## 執行流程
1. `python src/preprocess.py` → 把 Excel 轉 parquet
2. EDA 在 `notebooks/01_eda.ipynb`
3. 訓練 baselines 與 LightGCN：`python src/train.py`
4. 產生圖表與總表：`python src/visualize.py`
5. 產生乾淨版總表：`python -m src.metrics_summary`
6. 評估與視覺化在 `notebooks/03_visualization.ipynb`

## Demo 與部署
- 內部展示：`streamlit run app.py`，會顯示較完整的讀者資訊，只適合本機或受控環境。
- 公開展示：`streamlit run app_public.py`，不顯示真實讀者 ID，適合口試或公開 demo。
- API 展示：`uvicorn api.main:app --reload --port 8000`，首頁會載入 `api/static/index.html`。

## 評估指標
- Recall@K (K=10, 20)
- Precision@K
- NDCG@K
- HitRate@K
- MRR@K
- Coverage@K
- Novelty@K

`results/summary.csv` 是原始實驗總表；`results/summary_clean.csv` 會把缺漏欄位標成 `NA`，方便報告檢查。若某些模型的 Coverage/Novelty/MRR 是 `NA`，代表該模型需要用最新版 `src/evaluate.py` 重新評估，請勿手動補假數字。

## 測試
```bash
python -m pytest -q
```

目前測試以 smoke tests 為主，檢查評估指標、LightGCN shape、summary helper 與 API stats 邏輯。

## 環境
- Python 3.10
- PyTorch + PyTorch Geometric
- GPU: RTX 4060 8GB
