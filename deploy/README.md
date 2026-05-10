# 部署指南總覽

本資料夾包含把這個推薦系統部署成「網頁版」的多種方法。

## 比較表

| 方案 | 努力度 | 費用 | 公開？ | 速度 | 適用情境 |
|------|--------|------|--------|------|----------|
| **ngrok** | 5 min | 免費 | 是（但你電腦要開） | 你電腦的速度 | 口試臨時 demo |
| **Streamlit Cloud** | 15 min | 免費 | 是（永久） | 中等 (1 vCPU) | 履歷作品集 |
| **Hugging Face Spaces** | 15 min | 免費（CPU） | 是（永久） | 中等 | ML 圈標準展示 |
| **Docker (本機)** | 30 min | 免費 | 否（除非開 port） | GPU/CPU 看你機器 | 學 docker、給校內用 |
| **FastAPI + 自製前端** | 1-2 天 | 看部署方式 | 自由 | 自由 | 想練全端、做履歷 |

## 推薦學習路徑

1. **先試 ngrok**（最有成就感，5 分鐘看到公開 URL）
   - 看 [`ngrok_setup.md`](./ngrok_setup.md)
2. **再上 Hugging Face Spaces**（永久公開，履歷可放）
   - 看 [`huggingface_spaces.md`](./huggingface_spaces.md)
3. **(選做) FastAPI + 前端**（顯示全端能力）
   - 啟動：`uvicorn api.main:app --reload --port 8000`
   - 開瀏覽器：`http://localhost:8000`

---

## 公開展示安全規則

- 公開或口試展示請優先使用 `app_public.py` 或 `api/main.py`。
- `app.py` 是內部分析版，會顯示較完整的讀者資訊，只適合本機或受控環境。
- 不要把 `.env`、`data/raw/`、`data/processed/`、`data/splits/`、`checkpoints/` 直接上傳公開 repo；這些已列在 `.gitignore`。
- Demo 指標會從 `results/summary.csv` 讀取；若更新實驗結果，請重新執行 `python -m src.metrics_summary` 產生 `results/summary_clean.csv`。

---

## 我們有兩套 UI

| 檔案 | 框架 | 適合場景 |
|------|------|----------|
| `app_public.py` | Streamlit | 快速部署、免寫 HTML/CSS |
| `api/main.py` + `api/static/index.html` | FastAPI + Tailwind | 完全自訂的網頁、履歷加分 |

兩套都做了「隱私保護」設計：
- 不顯示真實讀者 ID
- 5 種 Persona 原型 demo
- 自訂喜歡的書 → 推薦

---

## 檔案說明

```
deploy/
├── README.md                  ← 你正在看
├── ngrok_setup.md            ← 5 分鐘公開 demo（最簡單）
├── start_ngrok.ps1           ← ngrok 一鍵啟動腳本
├── streamlit_cloud.md        ← 永久部署方案 1
├── huggingface_spaces.md     ← 永久部署方案 2
└── (Dockerfile 在專案根目錄)
```

---

## 常見問題

### Q1：我能直接給教授用 GitHub repo 連結嗎？
可以，但他需要自己安裝環境跑。**不如直接給網址**（用 ngrok 或 HF Spaces）。

### Q2：可以一次部署兩種 UI 嗎？
可以。Streamlit 跑在 8501、FastAPI 跑在 8000，互不衝突。
教授看哪個都一樣酷。

### Q3：要在履歷上放哪個？
- **HF Spaces 連結**：「ML 模型 demo」（短秒看到結果）
- **GitHub 連結**：「程式碼與報告」
- **FastAPI demo**：如果想找全端工作就放這個

### Q4：成本會很高嗎？
全部免費方案就夠用了（學生情境）。
若你的 demo 紅了被濫用、超過免費額度，再考慮付費（HF Spaces GPU 約 $0.6/小時）。

### Q5：原本的 `app.py` 還能用嗎？
可以。`app.py` 是「內部完整版」（顯示所有資訊、含真實 ID），給你自己用。
`app_public.py` 是「公開安全版」（隱私保護），給外人看。
