# run_qms.ps1 - Launcher for the ISO 9001:2015 QMS AI Auditor

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🛡️  STARTING ISO 9001:2015 QMS AI AUDITOR WEB APPLICATION  🛡️" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start the FastAPI backend server in the background
Write-Host "🚀 Launching FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Yellow
$BackendProcess = Start-Process python -ArgumentList "-m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000" -NoNewWindow -PassThru

# 2. Start the HTTP static server for the HTML/CSS/JS frontend in the background
Write-Host "🌐 Launching Vanilla Web Frontend on http://localhost:8080..." -ForegroundColor Yellow
$FrontendProcess = Start-Process python -ArgumentList "-m http.server 8080 --directory frontend" -NoNewWindow -PassThru

# 3. Wait for processes to spin up and load
Start-Sleep -Seconds 3

# 4. Open default web browser to the web frontend
Write-Host "🎯 Opening your Web Browser to http://localhost:8080..." -ForegroundColor Green
Start-Process "http://localhost:8080"

Write-Host ""
Write-Host "----------------------------------------------------------" -ForegroundColor Gray
Write-Host "App is running smoothly!" -ForegroundColor Green
Write-Host "To terminate both backend and frontend servers, press [Ctrl + C] or close this PowerShell window." -ForegroundColor Yellow
Write-Host "----------------------------------------------------------" -ForegroundColor Gray

# Maintain execution loop and handle termination signals
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "`nStopping servers..." -ForegroundColor Red
    
    # Terminate backend process
    if ($BackendProcess -and -not $BackendProcess.HasExited) {
        Write-Host "Stopping FastAPI Backend (PID: $($BackendProcess.Id))..." -ForegroundColor Red
        Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Terminate frontend process
    if ($FrontendProcess -and -not $FrontendProcess.HasExited) {
        Write-Host "Stopping Frontend Server (PID: $($FrontendProcess.Id))..." -ForegroundColor Red
        Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "Done! Goodbye." -ForegroundColor Green
}
