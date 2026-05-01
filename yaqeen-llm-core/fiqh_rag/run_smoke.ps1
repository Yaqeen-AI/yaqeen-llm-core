# Kill any Python process holding the Qdrant lock, then run the smoke test.
$lock = "$PSScriptRoot\qdrant_storage\.lock"
if (Test-Path $lock) {
    Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
    Write-Host "Qdrant lock cleared." -ForegroundColor Yellow
}
$env:PYTHONUTF8 = "1"
& "E:\PythonProject4\.venv\Scripts\python.exe" -m scripts.smoke_test
