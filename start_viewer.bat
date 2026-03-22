@echo off
echo ============================================================
echo Weather Radar Prediction System
echo ============================================================
echo.

if "%WEATHER_RADAR_VIEWER_PORT%"=="" set WEATHER_RADAR_VIEWER_PORT=5050
set TF_ENABLE_ONEDNN_OPTS=0

cd /d %~dp0
python start.py %*
