@echo off
rem Starts the whole system (fetcher, predictor, web viewer) with the project's virtualenv.
cd /d %~dp0
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. See README.md "Setup".
    pause
    exit /b 1
)
".venv\Scripts\python.exe" start.py %*
