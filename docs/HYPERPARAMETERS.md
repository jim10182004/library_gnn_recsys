# 超參數對照表

> 每個正式模型的完整實驗設定。對應論文 4.2.2 節「超參數網格搜尋」+ 4.2.3 節「Optuna 自動搜尋」。

---

## 共用基礎設定

下表所有模型若無特別說明，皆使用：

| 共用參數 | 值 | 說明 |
|---|---|---|
| `optimizer` | Adam | |
| `seed` | 42 | 主實驗用；multi-seed 用 42/123/2024 |
| `device` | cuda | RTX 4060 8GB；fallback CPU |
| `eval_every` | 5 | 每 5 epoch 評估 val |
| `early_stop_patience` | 10 | val R@20 連續 10 epoch 未進步 → 停 |
| `n_negatives` | 1 | BPR 每個 pos 抽 1 個 neg |
| `train period` | 2025-01 ~ 2025-10 | 時序切分 |
| `val period` | 2025-11 | |
| `test period` | 2025-12 | |
| `k-core` | ≥ 5 | 雙向過濾 |

---

## 一、Baseline 模型

| 模型 | embed_dim | n_layers | lr | batch_size | n_epochs | 備註 |
|---|---|---|---|---|---|---|
| `popular` | — | — | — | — | — | rule-based，不訓練 |
| `itemcf` | — | — | — | — | — | 共現矩陣，不訓練 |
| `bprmf` | 64 | — | 1e-3 | 4096 | 30 | 標準 BPR Matrix Factorization |

## 二、LightGCN 家族（主實驗）

| 模型 | embed_dim | n_layers | lr | batch_size | weight_decay | n_epochs | reserve_weight | 額外設定 |
|---|---|---|---|---|---|---|---|---|
| `lightgcn` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | — | |
| `lightgcn_si` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | — | side_info: gender+age+category |
| `lightgcn_multi` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | **0.5** | + side info |
| `lightgcn_bert` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | — | + BERT (multilingual MiniLM, 384d) |
| `lightgcn_hetero` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | — | + 作者節點異質圖 |
| `lightgcn_timedecay` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | 0.5 | decay_lambda=0.05 |
| `lightgcn_tgn` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | — | Time2Vec dim=8 |
| `lightgcn_cover` | 64 | 3 | 1e-3 | 4096 | 1e-4 | 50 | — | ResNet-18 cover, 512→64 |

## 三、對照組（前沿 SOTA）

| 模型 | embed_dim | n_layers | lr | batch_size | n_epochs | 特殊參數 |
|---|---|---|---|---|---|---|
| `ngcf` | 64 | 3 | 1e-3 | 4096 | 50 | message_dropout=0.1, node_dropout=0 |
| `simgcl` | 64 | 3 | 1e-3 | 4096 | 50 | **eps=0.02, cl_weight=0.001**（sweep 後最佳） |
| `sasrec` | 64 | 2 | 1e-3 | 256 | 100 | max_len=50, n_heads=2, dropout=0.2 |

> **SimGCL 重要**：原始預設 `eps=0.1, cl_weight=0.1` 在本資料災難性差（R@10=0.15）。
> 經 `src/simgcl_sweep.py` 調出 `eps=0.02, cl_weight=0.001` 才達 LightGCN 同等水準。

## 四、Optuna 調參版本（最終最佳）⭐

| 模型 | embed_dim | n_layers | lr | batch_size | weight_decay | n_epochs | 來源 |
|---|---|---|---|---|---|---|---|
| `lightgcn_opt` | **128** | **2** | **0.00281** | **2048** | **5.65e-5** | 50 | Optuna 20 trials |
| `lightgcn_multi_opt` ★ | **128** | **2** | **0.00281** | **2048** | **5.65e-5** | 50 | Optuna 套用至 Multi |

### Optuna 搜尋空間（src/optuna_search.py）

| 超參數 | 搜尋範圍 | 採樣方式 |
|---|---|---|
| `embed_dim` | {32, 64, 128} | categorical |
| `n_layers` | {1, 2, 3, 4} | categorical |
| `lr` | [5e-4, 5e-3] | log-uniform |
| `weight_decay` | [1e-7, 1e-3] | log-uniform |
| `batch_size` | {1024, 2048, 4096} | categorical |

- **Sampler**：TPE (Tree-structured Parzen Estimator)
- **Trials**：20
- **Objective**：Validation Recall@20
- **最佳 trial**：Val R@20 = 0.3233（vs grid 最佳 0.3034，+6.6%）

詳見 `results/ablation/optuna_best.json` 與 `results/ablation/optuna.csv`。

---

## 五、Ablation 實驗設定（多個變體）

### 5.1 Grid Search (embed_dim × n_layers)

| 變體 | embed_dim | n_layers | 其他 |
|---|---|---|---|
| `lightgcn_d32_L1` ~ `lightgcn_d128_L4` | {32,64,128} | {1,2,3,4} | lr=1e-3, batch=4096, epochs=50 |

12 個組合，對應 `results/ablation/hyperparam.csv`。

### 5.2 Reserve Weight Sweep

| 變體 | reserve_weight | 其他 |
|---|---|---|
| `lgcn_multi_rw00` | 0.0 | 等同於 `lightgcn_si` |
| `lgcn_multi_rw03` | 0.3 | |
| `lgcn_multi_rw05` | 0.5 | 主實驗 baseline |
| `lgcn_multi_rw07` | 0.7 | |
| `lgcn_multi_rw10` | 1.0 | 預約等同借閱 |

5 個變體，其他超參數固定為 `lightgcn_multi` 預設。

### 5.3 Side Info 啟用組合

| 變體 | gender | age | category |
|---|---|---|---|
| `lgcn_si_none` | ✗ | ✗ | ✗ |
| `lgcn_si_gender_only` | ✓ | ✗ | ✗ |
| `lgcn_si_age_only` | ✗ | ✓ | ✗ |
| `lgcn_si_category_only` | ✗ | ✗ | ✓ |
| `lgcn_si_g+a` | ✓ | ✓ | ✗ |
| `lgcn_si_g+c` | ✓ | ✗ | ✓ |
| `lgcn_si_a+c` | ✗ | ✓ | ✓ |
| `lgcn_si_all` | ✓ | ✓ | ✓ |

8 個組合，其他固定為 `lightgcn_si` 預設。

### 5.4 Multi-seed 實驗

| 模型 | seeds |
|---|---|
| `lightgcn` | 42, 123, 2024 |
| `lightgcn_si` | 42, 123, 2024 |
| `lightgcn_multi` | 42, 123, 2024 |

每組 3 次重複訓練，計算 mean ± std。對應 `results/ablation/multi_seed.csv`。

---

## 六、超參數選擇的「原因」

| 超參數 | 我選的值 | 為什麼 |
|---|---|---|
| `embed_dim = 128` | Optuna 結果 | 比預設 64 好 6.6%；128 表達力夠且不過擬合 |
| `n_layers = 2` | Optuna 結果 | 1 層太淺、4 層 over-smoothing；2 最平衡 |
| `lr = 0.00281` | Optuna 結果 | 比預設 1e-3 大 ~3 倍；配 batch=2048 收斂深 |
| `batch_size = 2048` | Optuna 結果 | 比預設 4096 小一半，gradient 訊號更密集 |
| `weight_decay = 5.65e-5` | Optuna 結果 | 輕微 L2 防過擬合，又不會把 embedding 壓成 0 |
| `n_epochs = 50` + early stop | 預設 | early stop 通常 30-40 觸發 |
| `n_negatives = 1` | 經驗值 | 提到 5 沒明顯改善但慢 5× |
| `reserve_weight = 0.5` | 主實驗 | 也試了 0/0.3/0.7/1.0，差距小，0.5 最穩 |

完整超參數理由見白話版第 1023-1051 行（`docs/白話版技術說明.md` 第 5 節「每個超參數做什麼」）。

---

## 重新跑某個模型的指令範本

```bash
# 1. 主實驗 LightGCN-Multi-Opt（最佳）
python -m src.train --model lightgcn_multi \
    --embed-dim 128 --n-layers 2 \
    --lr 0.00281 --batch-size 2048 --decay 5.65e-5 \
    --epochs 50 --reserve-weight 0.5 \
    --tag opt --seed 42

# 2. Ablation：reserve_weight = 0.7
python -m src.train --model lightgcn_multi \
    --reserve-weight 0.7 --tag rw07

# 3. Side info ablation：只用 category
python -m src.train --model lightgcn_si \
    --side-info category --tag category_only

# 4. Multi-seed
for seed in 42 123 2024; do
    python -m src.train --model lightgcn_multi --seed $seed --tag seed${seed}
done

# 5. Optuna 搜尋
python src/optuna_search.py --n-trials 20

# 6. SimGCL sweep
python src/simgcl_sweep.py
```
