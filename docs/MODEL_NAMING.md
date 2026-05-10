# 模型命名對照表

> 本研究的 16 個模型 + 中間 ablation 變體用了多種命名前綴。這份是統一速查。

---

## 一、最終 16 個正式模型（出現在論文 + summary.csv）

| 編號 | 命名 | checkpoint 檔案 | 中文簡稱 | 備註 |
|---|---|---|---|---|
| 1 | `popular` | （無，rule-based） | 熱門基線 | 推全民熱門書 |
| 2 | `itemcf` | （無，rule-based） | 物品協同過濾 | 2001 年經典 |
| 3 | `bprmf` | `bprmf_best.pt` | BPR Matrix Factorization | 2009 經典神經網路 |
| 4 | `lightgcn` | `lightgcn_best.pt` | **基本 LightGCN** | SIGIR 2020 |
| 5 | `lightgcn_si` | `lightgcn_si_best.pt` | + Side Info（性別/年齡/分類） | |
| 6 | `lightgcn_multi` | `lightgcn_multi_best.pt` | + 預約 weak edges | |
| 7 | `lightgcn_bert` | `lightgcn_bert_best.pt` | + BERT 書名語意 | multilingual MiniLM |
| 8 | `lightgcn_hetero` | `lightgcn_hetero_best.pt` | + 作者節點異質圖 | |
| 9 | `lightgcn_timedecay` | `lightgcn_timedecay_best.pt` | + 時間衰減邊權重 | |
| 10 | `lightgcn_tgn` | `lightgcn_tgn_best.pt` | + Time2Vec 時間編碼 | TGN-style |
| 11 | `lightgcn_cover` | `lightgcn_cover_best.pt` | + ResNet-18 書封 feature | 多模態 PoC |
| 12 | `ngcf` | `ngcf_best.pt` | LightGCN 前身（複雜版） | SIGIR 2019 |
| 13 | `simgcl` | `simgcl_best.pt` | 對比學習 | SIGIR 2022 |
| 14 | `sasrec` | `sasrec_best.pt` | Transformer 序列模型 | ICDM 2018 |
| 15 | `lightgcn_opt` | `lightgcn_opt_best.pt` | LightGCN + Optuna 調參 | embed=128, lr=0.0028 |
| 16 | `lightgcn_multi_opt` | `lightgcn_multi_opt_best.pt` | **★ 最佳模型** | Multi + Optuna |

## 二、Multi-seed 變體（驗證穩定度，不在論文主表）

| 命名規則 | 檔案範例 | 用途 |
|---|---|---|
| `lightgcn_seed{N}` | `lightgcn_seed42_best.pt`、`...seed123_best.pt`、`...seed2024_best.pt` | 算 mean ± std |
| `lightgcn_si_seed{N}` | `lightgcn_si_seed42_best.pt`、... | 同上 |
| `lightgcn_multi_seed{N}` | `lightgcn_multi_seed42_best.pt`、... | 同上 |

對應 `results/ablation/multi_seed.csv`。

## 三、超參數 grid 變體（4.2.2 節）

| 命名規則 | 檔案範例 | embed_dim | n_layers |
|---|---|---|---|
| `lightgcn_d{D}_L{L}` | `lightgcn_d32_L1_best.pt` | 32 | 1 |
| | `lightgcn_d32_L2_best.pt` | 32 | 2 |
| | ... | ... | ... |
| | `lightgcn_d128_L4_best.pt` | 128 | 4 |

12 種組合（embed_dim ∈ {32,64,128} × n_layers ∈ {1,2,3,4}），對應 `results/ablation/hyperparam.csv`。

## 四、Reserve weight 變體（4.2.3 節）

| 命名 | reserve_weight 值 | 含意 |
|---|---|---|
| `lgcn_multi_rw00` | 0.0 | 不用預約 |
| `lgcn_multi_rw03` | 0.3 | 預約權重 0.3 |
| `lgcn_multi_rw05` | 0.5 | 預約權重 0.5 |
| `lgcn_multi_rw07` | 0.7 | 預約權重 0.7 |
| `lgcn_multi_rw10` | 1.0 | 預約等同借閱 |

對應 `results/ablation/reserve_weight.csv`。

## 五、Side-info ablation 變體（4.2.4 節）

| 命名 | 啟用的 side info | 含意 |
|---|---|---|
| `lgcn_si_none` | （無） | 對照組 |
| `lgcn_si_gender_only` | gender | |
| `lgcn_si_age_only` | age | |
| `lgcn_si_category_only` | category | |
| `lgcn_si_g+a` | gender + age | |
| `lgcn_si_g+c` | gender + category | |
| `lgcn_si_a+c` | age + category | |
| `lgcn_si_all` | 全部 | 完整版 |

對應 `results/ablation/side_info.csv`。

---

## 命名規則總結

| 前綴 | 含意 | 出現位置 |
|---|---|---|
| `lightgcn_` | 完整正式模型 | 論文主表 |
| `lgcn_` | Ablation 變體（短名稱） | results/ablation/ |
| `_seed{N}` | Multi-seed 重複實驗 | 穩定度檢驗 |
| `_d{D}_L{L}` | 超參數 grid 變體 | hyperparam ablation |
| `_rw{W}` | reserve_weight 變體 | reserve weight ablation |
| `_opt` | Optuna 調出的超參數 | 最終最佳版本 |
| `_best.pt` | 該模型的最佳 checkpoint | 所有 checkpoint 都有 |

## 對應關係圖

```
LightGCN 家族
├─ lightgcn (base)                    ─ 64 dim, 3 layers, lr=1e-3
│  ├─ lightgcn_seed{42|123|2024}      (multi-seed)
│  └─ lightgcn_d{32|64|128}_L{1|2|3|4} (12 個 grid 變體)
├─ lightgcn_si (+ side info)
│  ├─ lightgcn_si_seed{...}
│  └─ lgcn_si_{none|gender_only|...|all} (8 個 ablation)
├─ lightgcn_multi (+ 預約)
│  ├─ lightgcn_multi_seed{...}
│  └─ lgcn_multi_rw{00|03|05|07|10}   (5 個 reserve weight)
├─ lightgcn_bert (+ BERT)
├─ lightgcn_hetero (+ 作者節點)
├─ lightgcn_timedecay (+ 時間衰減)
├─ lightgcn_tgn (+ Time2Vec)
├─ lightgcn_cover (+ 書封 CNN)
└─ Optuna 系列
   ├─ lightgcn_opt          (純 ID + Optuna)
   └─ lightgcn_multi_opt    ★ 最佳

對照組
├─ ngcf (LightGCN 前身)
├─ simgcl (對比學習)
└─ sasrec (序列模型)

Baselines
├─ popular
├─ itemcf
└─ bprmf
```

## Quick lookup：我看到 X，那是什麼？

- `lgcn_si_g+c_best.pt` → side info ablation, 啟用 gender + category, 對應 `lgcn_si_g+c_history.json`
- `lgcn_multi_rw07_best.pt` → reserve_weight=0.7 的多邊型版
- `lightgcn_d128_L2_best.pt` → embed_dim=128, n_layers=2 的 grid 點（也就是 best grid 設定）
- `lightgcn_multi_seed123_best.pt` → multi-seed 實驗的 seed=123 版本
- `lightgcn_opt_best.pt` → Optuna 找出的最佳純 ID LightGCN
- `lightgcn_multi_opt_best.pt` → ★ 最終最佳模型
