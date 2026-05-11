---
title: Library GNN Recommendation
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: LightGCN book recommendation - polished UI
---

# 📚 圖書館 GNN 推薦系統 — 完整 UI 版

基於 **LightGCN** 的個人化書籍推薦系統，以 **FastAPI + Tailwind CSS + Chart.js** 自製前端。

## 互動功能

- 🎭 **11 個人物原型**：點即看推薦
- ✏️ **自訂模式**：搜尋你看過的書 → 即時推薦
- 🆚 **比較模式**：兩個 persona 並排，重疊書黃框標出
- ✨ **MMR 重排序**：可切換多樣性版本
- 💬 **「為什麼推薦」彩色 badge**：作者相同 / 同類別 / 相似讀者群
- 📊 **分類分布圖**：Chart.js 即時呈現

## 技術棧

- **Backend**: FastAPI + uvicorn
- **Model**: LightGCN（預計算 embedding）
- **Frontend**: Tailwind CSS + Chart.js（單檔 self-contained HTML）
- **Inference**: CPU cosine similarity (~17 ms / req)

## 隱私

✅ 本 demo **完全不含讀者資訊**
- ❌ 無讀者 ID / 借閱事件 / demographics
- ✅ 僅書本公開 metadata + 預訓練 64 維 embedding

## 完整研究

- **GitHub**：https://github.com/jim10182004/library_gnn_recsys
- 16 模型對照、Optuna 自動調參、multi-seed 驗證、論文 + 簡報 + 白話版說明
