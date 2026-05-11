# Demo Bundle

本資料夾為 HuggingFace Spaces 部署用的最小資料包，**不包含任何讀者資訊**。

## 檔案說明

| 檔案 | 大小 | 內容 |
|---|---|---|
| `item_embs.pt` | ~8 MB | 預計算的書本 64 維 embedding（LightGCN propagate 結果）|
| `books_meta.parquet` | ~9 MB | 書本 metadata：title, author, category, ISBN, pub_year |
| `item_remap.json` | ~500 KB | 原始 book_id → compact_id 對應表 |
| `metadata.json` | < 1 KB | bundle 版本與模型資訊 |

## 隱私聲明

- ❌ **無**讀者 ID
- ❌ **無**借閱事件 (borrows)
- ❌ **無**預約事件 (reservations)
- ❌ **無**讀者 demographics (gender / age)
- ✅ **僅**書本公開 metadata（書名/作者等公開資訊）+ embedding 向量

embedding 向量是模型訓練後的「書本身份卡」（64 個浮點數），無法回推任何讀者借閱歷史。

## 重新生成

```bash
python deploy/hf_spaces/build_bundle.py
```
