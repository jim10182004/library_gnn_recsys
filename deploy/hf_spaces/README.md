---
title: Library GNN Recommendation Demo
emoji: 📚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
short_description: GNN-based book recommendation (LightGCN) on a city public library's borrowing data
---

# 📚 圖書館 GNN 推薦系統 Demo

基於 **LightGCN** 的個人化書籍推薦系統，部署在 HuggingFace Spaces 的純 CPU 互動 demo。

## 特色

- 🧠 **真實的 GNN 模型推薦**（不是 fake/random，是訓練好的 LightGCN）
- 🎭 **11 個人物原型**：點選即看推薦結果
- ✏️ **自訂模式**：輸入 3-5 本你看過的書，模型即時給出 10 本類似書
- 🆚 **比較模式**：並排兩個 persona，重疊書本黃框標出
- ✨ **MMR 重排序**：可切換多樣性版本（同類別最多 6 本）

## 隱私

本 demo **完全不包含讀者資訊**：
- ❌ 無讀者 ID
- ❌ 無借閱事件
- ❌ 無讀者 demographics
- ✅ 僅書本公開 metadata + 預訓練的 64 維書本向量（embedding）

## 完整研究

- **GitHub**：https://github.com/jim10182004/library_gnn_recsys
- **論文** + **簡報** + **白話版說明** 都在 `docs/` 資料夾
- 16 個模型對照、Optuna 自動調參、multi-seed 統計顯著性檢驗

## 重新生成 demo bundle

```bash
python deploy/hf_spaces/build_bundle.py
```

會把訓練好的 model + books metadata 打包成 ~10 MB 的 self-contained bundle。
