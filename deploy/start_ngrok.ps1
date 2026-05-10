# 一鍵啟動 Streamlit + ngrok，取得公開 URL（適合口試臨時 demo）
#
# 使用前準備：
#   1. 安裝 ngrok：https://ngrok.com/download （解壓到 PATH）
#   2. 註冊免費帳號取得 token：https://dashboard.ngrok.com/get-started/your-authtoken
#   3. 一次性設定 token：ngrok config add-authtoken YOUR_TOKEN
#
# 執行：
#   .\deploy\start_ngrok.ps1
#
# 結束：Ctrl+C 兩次（先關 ngrok，再關 streamlit）

$ErrorActionPreference = "Stop"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  圖書館 GNN 推薦系統 - 公開 Demo 啟動" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 檢查 ngrok
if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 找不到 ngrok，請先安裝：https://ngrok.com/download" -ForegroundColor Red
    exit 1
}

# 啟動 Streamlit (background)
Write-Host "[1/2] 啟動 Streamlit 在 port 8501 ..." -ForegroundColor Yellow
$streamlit = Start-Process -FilePath "streamlit" `
    -ArgumentList "run", "app_public.py", "--server.port=8501", "--server.headless=true" `
    -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 5

# 啟動 ngrok（前景，這樣 Ctrl+C 會關它）
Write-Host "[2/2] 啟動 ngrok tunnel ..." -ForegroundColor Yellow
Write-Host ""
Write-Host "公開 URL 將顯示在下方（Forwarding 那行）" -ForegroundColor Green
Write-Host "把 https://...ngrok.io 那個網址複製給教授即可" -ForegroundColor Green
Write-Host ""
Write-Host "結束：在此視窗按 Ctrl+C，再關 streamlit (PID=$($streamlit.Id))" -ForegroundColor Yellow
Write-Host ""

try {
    ngrok http 8501
} finally {
    Write-Host "[Cleanup] 關閉 streamlit..." -ForegroundColor Yellow
    Stop-Process -Id $streamlit.Id -Force -ErrorAction SilentlyContinue
}
