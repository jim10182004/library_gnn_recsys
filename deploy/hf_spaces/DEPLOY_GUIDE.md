# HuggingFace Spaces 部署完整步驟

> 從 GitHub repo → 線上公開 demo URL，總共 5 分鐘

---

## 你會得到什麼

部署完成後，會有一個公開 URL 像這樣：
```
https://huggingface.co/spaces/jim10182004/library-gnn-demo
```

任何人開瀏覽器都能玩 11 個 persona、自訂模式、比較模式。**免費 + 不需信用卡 + 永久線上**。

---

## 步驟 0：先確定 GitHub 已 push（你已完成）

```bash
git remote -v
# 應該看到 origin → github.com/jim10182004/library_gnn_recsys.git
```

✅ 已完成（commit `c7d0e06` 之後）

---

## 步驟 1：到 HuggingFace 註冊 / 登入

→ https://huggingface.co/join （免費，可用 Google / GitHub 一鍵登入）

完成後拿到你的 username（例如 `jim10182004`）。

---

## 步驟 2：建立一個新 Space

→ https://huggingface.co/new-space

填這幾個欄位：

| 欄位 | 填什麼 |
|---|---|
| Space name | `library-gnn-demo` |
| License | `mit` |
| Select the Space SDK | **Streamlit** |
| Hardware | **CPU basic (free)** |
| Visibility | Public |

按 `Create Space`。HF 會建立一個空的 git repo。

---

## 步驟 3：本機把 deploy bundle push 到那個 Space

在 PowerShell（或 Git Bash）：

```bash
# 1. clone 你新建的 Space repo（HF 會給你一個新的 git URL）
cd $HOME/Downloads
git clone https://huggingface.co/spaces/jim10182004/library-gnn-demo
cd library-gnn-demo

# 2. 把 deploy bundle 全部複製過來
cp -r "$HOME/OneDrive - 東吳大學/桌面/library_gnn_recsys/deploy/hf_spaces/"* .

# 3. push（首次會問 HF token，到 https://huggingface.co/settings/tokens 產生 write token）
git add -A
git commit -m "Initial deployment from library_gnn_recsys"
git push
```

> **提示**：HF push 時要輸入 **HuggingFace token**，不是 password。
> 到 https://huggingface.co/settings/tokens 點 `New token` → 給 `write` 權限 → 貼進來。

---

## 步驟 4：等 HF Spaces 自動 build（約 2-3 分鐘）

回到瀏覽器，打開 `https://huggingface.co/spaces/jim10182004/library-gnn-demo`，會看到：

- 第 1 階段：`Building` （顯示 logs）
- 第 2 階段：`Running` （正常）

如果出錯，logs 會有清楚的錯誤訊息。

---

## 步驟 5：分享給朋友 / 教授 / 雇主

把這個 URL 放進：
- LinkedIn / 履歷
- GitHub repo README（已自動連結）
- 口試簡報 PPT 第 18 頁

---

## 之後要更新怎麼辦？

每次重新訓練模型後：

```bash
# 1. 在 library_gnn_recsys 重新生成 bundle
cd "$HOME/OneDrive - 東吳大學/桌面/library_gnn_recsys"
python deploy/hf_spaces/build_bundle.py

# 2. 複製到 HF Space repo
cp -r deploy/hf_spaces/* "$HOME/Downloads/library-gnn-demo/"

# 3. push
cd "$HOME/Downloads/library-gnn-demo"
git add -A && git commit -m "Update model" && git push
```

---

## 替代方案：Streamlit Cloud（如果偏好）

Streamlit Cloud（streamlit.io）也支援，更直接：

1. 到 https://share.streamlit.io/ 用 GitHub 登入
2. New app → 選 `jim10182004/library_gnn_recsys` repo
3. **Main file path**: `deploy/hf_spaces/app.py`
4. Deploy

**差異**：Streamlit Cloud 直接從 GitHub repo 讀，不需要 push 到別的地方。
但 free tier 有 1 GB RAM 限制，要看模型大小。

---

## Troubleshooting

| 症狀 | 解法 |
|---|---|
| HF Space build 失敗 | 看 logs，通常是 `requirements.txt` 版本問題 |
| 「No module named torch」 | requirements.txt 第 1 行少了 `torch>=2.0` |
| 載入很慢 | CPU basic 是 2 vCPU + 16GB RAM，第一次載入會慢 30 秒 |
| HF push 被拒（401） | Token 沒設 `write` 權限，到 settings/tokens 改 |
| `assets/item_embs.pt` 太大 | HF 支援 git-lfs，但 8 MB 還在普通限制內，不用 lfs |
