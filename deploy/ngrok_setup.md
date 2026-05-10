# ngrok 一鍵 demo 指南（最快、最適合口試）

ngrok 把你電腦的 localhost 變成公開 URL，**0 部署、0 雲端帳號**。

**最佳使用情境**：口試當下，教授點連結直接看 demo。

---

## 一次性設定（5 分鐘）

### 1. 下載 ngrok
[https://ngrok.com/download](https://ngrok.com/download)

下載 Windows 版 zip → 解壓到任何資料夾，例如 `C:\tools\ngrok`

### 2. 加到 PATH
PowerShell（系統管理員）執行：
```powershell
[System.Environment]::SetEnvironmentVariable('Path', $env:Path + ';C:\tools\ngrok', 'User')
```

重新打開 PowerShell，測試：
```powershell
ngrok --version
```

### 3. 註冊 ngrok 帳號取得 authtoken
[https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)

註冊後到 [authtoken 頁面](https://dashboard.ngrok.com/get-started/your-authtoken) 複製你的 token。

### 4. 設定 token
```powershell
ngrok config add-authtoken YOUR_TOKEN_HERE
```

---

## 啟動 demo

### 方法 1：用我寫的腳本（推薦）
```powershell
cd "C:\Users\USER\OneDrive - 東吳大學\桌面\library_gnn_recsys"
.\deploy\start_ngrok.ps1
```

會自動啟動 streamlit + ngrok，畫面會顯示像這樣：
```
Forwarding   https://abc123.ngrok-free.app -> http://localhost:8501
```

把那個 `https://abc123.ngrok-free.app` 給教授就 OK 了。

### 方法 2：手動
開兩個 PowerShell 視窗：

視窗 A：
```powershell
cd "C:\Users\USER\OneDrive - 東吳大學\桌面\library_gnn_recsys"
streamlit run app_public.py --server.port=8501
```

視窗 B：
```powershell
ngrok http 8501
```

---

## 重要提醒

### 你電腦不能關機 / 睡眠
ngrok URL 只在你的 streamlit 在跑時才有效。
- 口試時：**用插電 + 關掉自動睡眠**
- 結束口試後可以關閉，URL 自然失效

### 免費版的限制
- 每次啟動 URL 都會變（除非升級到付費版可固定 URL）
- 同時連線數 40（夠教授看 demo）
- 8 hours 連線 timeout（重啟即可）

### 安全性
- 任何人有 URL 就能用 → 不要長時間開著
- 我們的 demo 已做隱私保護（沒有真實 ID 顯示），所以還算安全
- 不要分享 URL 到公開網路（Twitter、PTT 等）

---

## 替代方案：Cloudflare Tunnel

如果不想用 ngrok，也可以用 [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/install-and-setup/tunnel-guide/)。設定稍複雜但更穩定、URL 可固定。
