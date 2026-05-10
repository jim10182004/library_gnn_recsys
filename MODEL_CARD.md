# Model Card — LightGCN-Multi-Opt

> 依 [Mitchell et al., 2019 "Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993) 標準格式。

---

## 模型概要 (Overview)

| 欄位 | 內容 |
|---|---|
| 模型名稱 | **LightGCN-Multi-Opt** |
| 模型版本 | v1.0（2026-05） |
| 模型類別 | Graph Neural Network 推薦系統 |
| 基底論文 | [LightGCN (He et al., SIGIR 2020)](https://arxiv.org/abs/2002.02126) |
| 訓練資料 | 某市立圖書館 2025 年 1-10 月借閱（45.4 萬筆，K-core ≥ 5 過濾後）+ 預約 weak edges |
| 訓練時間 | 約 30 分鐘（RTX 4060 8GB） |
| 模型大小 | ~30 MB（embedding 為主） |
| 推論延遲 | ~17 ms / request（GPU） |

## 預期用途 (Intended Use)

### 主要用途
- **個人化書籍推薦**：給定讀者 ID（或 4 本喜歡的書），輸出 Top-K 推薦書單
- **冷啟動 fallback**：透過 11 個 Persona 處理新讀者（無借閱史）
- **學術研究**：作為「LightGCN 在中文圖書館資料」的對照 baseline

### 適用場景
- ✅ 公共圖書館 OPAC「猜你會喜歡」推薦欄
- ✅ 圖書館員選書參考
- ✅ 學術畢業專題教學示範
- ✅ 推薦系統教學案例

### 非預期用途
- ❌ 商業電商推薦（資料分布不同）
- ❌ 使用者心理畫像或廣告投放
- ❌ 重新識別匿名讀者（**嚴禁**）
- ❌ 取代圖書館員的書單規劃（建議作為輔助而非取代）

## 訓練資料

詳見 [DATA_CARD.md](DATA_CARD.md)。摘要：
- **時序切分**：1-10 月 train、11 月 val、12 月 test
- **K-core ≥ 5**：保留至少 5 次互動的讀者與書本
- **規模**：3.6 萬讀者 × 3 萬書 × 45.4 萬借閱 + 1.16 萬額外預約 weak edges

## 模型架構

```
讀者 embedding (3.6 萬 × 128) ─┐
                               ├──▶ LightGCN 卷積 × 2 層 ──▶ 內積 ──▶ Top-K
書本 embedding (3 萬 × 128)  ─┘                                ↑
                                                              │
                  side_info（性別/年齡/分類）─────── 加入 ──┘
```

- **超參數**（Optuna 搜尋找出）：
  - `embed_dim = 128`
  - `n_layers = 2`
  - `lr = 0.00281`
  - `batch_size = 2048`
  - `weight_decay = 5.65e-5`
  - `n_epochs = 50`（含 early stopping，patience=10）

- **損失函數**：BPR loss with negative sampling (n_neg=1)
- **優化器**：Adam
- **GNN normalization**：對稱正規化 D^(-1/2) A D^(-1/2)

## 性能指標

在 12 月 test set（30,067 筆借閱）上的成績：

| 指標 | 數值 | 比 LightGCN baseline 提升 |
|---|---|---|
| Recall@10 | **0.2707** | +2.2% |
| Recall@20 | 0.3015 | +1.3% |
| NDCG@10 | 0.2232 | +2.5% |
| Hit@10 | 0.4307 | +2.3% |
| **Coverage@10** | **0.2651** | **+345%** ⭐ |
| Novelty@10 | 0.3994 | +27% |
| MRR@10 | 0.2787 | +2.6% |

### 與其他方法比較（Recall@10）

| 模型 | Recall@10 | 相對 LightGCN-Multi-Opt |
|---|---|---|
| Popular（基線） | 0.2532 | -6.5% |
| BPR-MF | 0.2544 | -6.0% |
| NGCF | 0.2639 | -2.5% |
| LightGCN | 0.2648 | -2.2% |
| LightGCN-BERT | 0.2674 | -1.2% |
| **LightGCN-Multi-Opt** | **0.2707** | — |

### 統計顯著性
- LightGCN → Multi: paired t-test p < 0.005（n=3 seeds）
- 所有主要差距 std < 1/10 of mean diff → 統計顯著

## 公平性分析

| 群組 | Recall@10 | 與全體差距 |
|---|---|---|
| 男性讀者 | 0.267 | -1.5% |
| 女性讀者 | 0.270 | +0.0% |
| < 18 歲 | 0.233 | -14.0% |
| 18-24 歲 | 0.321 | +18.9% |
| 35-49 歲 | 0.268 | -0.7% |

- ✅ **性別公平**：男女差距僅 1%
- ⚠️ **年齡有偏差但可解釋**：兒童書類別狹窄、大學生借閱主題明確
- 詳見 [results/fairness.md](results/fairness.md)

## 已知限制

1. **冷啟動讀者**：對全新讀者（無借閱史）需要 Persona fallback
2. **過於活躍讀者**：借閱 51+ 本的讀者 R@10 僅 0.03（個人化越深越難）
3. **時間範圍**：僅訓練 2025 年資料，未驗證跨年趨勢
4. **語言**：BERT/Cover 對中文書封資料源覆蓋不足（Open Library 4.4%）
5. **離線評估**：未做線上 A/B 測試，實際使用者點擊行為可能與離線不同
6. **GPU 記憶體**：8GB 上限，無法跑 d=256 或 batch=16384

## 倫理考量

- **隱私**：所有讀者 ID 已匿名化；Demo 不接受真實 ID 輸入
- **同意**：原始借閱資料的隱私處理由圖書館負責；本研究僅在已脫敏的資料上工作
- **偏見**：本模型可能反映歷史借閱偏好（例如某類書被某性別/年齡較多人借），不應用於人事或保險決策
- **再現性**：所有程式碼公開，實驗可一行重現

## 重新訓練 / 部署建議

```bash
# 重新訓練（GPU，~30 分鐘）
python -m src.train --model lightgcn_multi --embed-dim 128 --n-layers 2 \
    --lr 0.00281 --batch-size 2048 --decay 5.65e-5 --epochs 50 \
    --reserve-weight 0.5 --tag opt
```

## 引用

```bibtex
@misc{library_gnn_optune_2026,
  title  = {LightGCN-Multi-Opt: Optuna-Tuned LightGCN with Reservation Edges
            for Public Library Book Recommendation},
  author = {[作者姓名]},
  year   = {2026},
  note   = {Graduation project, Soochow University},
}
```

主要參考：
- He, Deng, Wang, Li, Zhang, Wang. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. SIGIR 2020.

## 聯絡

問題回報請至 GitHub issue 或聯繫作者。
