@echo off
echo Stopping any existing Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo Clearing Qdrant lock files...
if exist qdrant_storage\.lock del /F /Q qdrant_storage\.lock >nul 2>&1
if exist qdrant_cache\.lock   del /F /Q qdrant_cache\.lock   >nul 2>&1
if exist data\.lock           del /F /Q data\.lock           >nul 2>&1

echo Starting FiqhRAG...
python app.py
