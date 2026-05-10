# 模型權重與資料下載

## 為什麼這些檔案不在 git 裡？

| 類型 | 大小 | 為什麼不追蹤 |
|---|---|---|
| `data/raw/*.xlsx` | ~160 MB | 含敏感資訊（即使匿名），授權僅內部用 |
| `data/processed/*.parquet` | ~32 MB | 同上，且可由原始資料重建 |
| `data/splits/*.parquet` | ~10 MB | 可由 `python -m src.dataset` 重建 |
| `checkpoints/*.pt` | ~30 MB × 30 = ~900 MB | 太大，且可重新訓練 |
| `results/figures/*.png` | ~10 MB | 可由 `python src/visualize.py` 重新產生 |
| `results/summary.csv` | ~30 KB | 可由 `python -m src.recompute_full_metrics` 重建 |

## 三種取得方式

### 方式 1：完全自己跑（推薦）

```bash
# 1. 取得原始 Excel（需自己的圖書館合作 / 或修改成你的資料）
# 2. 設定 .env：複製 .env.example，填寫 LIBRARY_RAW_DIR
cp .env.example .env
# 3. 一鍵全部跑（GPU 約 3-4 小時）
python run_all.py
```

完成後會產生：
- `data/processed/*.parquet`（前處理結果）
- `data/splits/*.parquet`（切分）
- `checkpoints/*.pt`（30 個模型權重）
- `results/figures/*.png`、`results/summary.csv`、`results/*.json`

### 方式 2：只跑特定模型

```bash
# 只跑最佳模型
python -m src.train --model lightgcn_multi --embed-dim 128 --n-layers 2 \
    --lr 0.00281 --batch-size 2048 --decay 5.65e-5 --epochs 50 \
    --reserve-weight 0.5 --tag opt
```

### 方式 3：取得已訓練檔案

本專案**不公開散布**權重檔（避免被誤用為商業推薦或外洩讀者偏好）。

如果你是：
- 教授口試委員：可請作者提供
- 同校學弟妹想參考：請聯繫作者，並簽具學術用途同意書
- 其他：請從方式 1 自己跑

聯絡方式：見 README.md 末

## 「我的 demo 跑不起來」常見排錯

### `FileNotFoundError: data/processed/books.parquet`
你還沒前處理。先跑 `python src/preprocess.py`。

### `FileNotFoundError: checkpoints/lightgcn_best.pt`
還沒訓練。先跑 `python -m src.train --model lightgcn`。

### CUDA out of memory
- 改用 CPU：`--device cpu`（會慢約 10×）
- 縮 batch：`--batch-size 1024` 或 `512`
- 縮 embedding：`--embed-dim 32`

### Windows pyarrow + torch DLL 衝突
本專案所有 entry script 已處理（先 `import pandas, pyarrow` 再 `import torch`）。
如果你寫新 script，記得遵守此順序。

## 檔案大小速查

| 路徑 | 大小 | 是否在 git | 怎麼取得 |
|---|---|---|---|
| `data/raw/borrows.xlsx` | 100 MB | ❌ | 圖書館授權 |
| `data/processed/borrows.parquet` | 19 MB | ❌ | `python src/preprocess.py` |
| `data/splits/train.parquet` | 8 MB | ❌ | `python -m src.dataset` |
| `checkpoints/lightgcn_best.pt` | ~10 MB | ❌ | `python -m src.train --model lightgcn` |
| `checkpoints/lightgcn_multi_opt_best.pt` | ~22 MB | ❌ | `python -m src.train --model lightgcn_multi --tag opt ...` |
| `results/summary.csv` | 6 KB | ❌ | `python src/visualize.py` |
| `docs/論文_完整版.docx` | 2.3 MB | ✅ | 已追蹤 |
| `docs/圖書館GNN推薦系統_簡報.pptx` | 940 KB | ✅ | 已追蹤 |
| `docs/白話版技術說明.docx` | 100 KB | ✅ | 已追蹤 |
| `src/**/*.py` | ~150 KB | ✅ | 已追蹤 |
| `tests/**/*.py` | ~20 KB | ✅ | 已追蹤 |

## 估算磁碟需求

完整跑完一次 `run_all.py` 需要：

| 用途 | 大小 |
|---|---|
| 原始 Excel | 160 MB |
| Processed parquet | 32 MB |
| Splits | 10 MB |
| BERT cache | ~120 MB（一次性下載） |
| 30 個 checkpoints | 約 900 MB |
| Figures + JSON results | 30 MB |
| **總計** | **~1.3 GB** |

請預留至少 2 GB 磁碟空間。
