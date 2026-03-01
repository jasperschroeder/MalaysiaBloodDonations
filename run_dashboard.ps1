# PowerShell script to launch the Streamlit dashboard
Write-Host "Starting Malaysia Blood Donations Dashboard..." -ForegroundColor Green
Write-Host "Dashboard will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONPATH = "$PSScriptRoot\src"
& "$PSScriptRoot\.venv\Scripts\streamlit.exe" run "$PSScriptRoot\src\dashboard.py"
