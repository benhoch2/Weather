@echo off
echo ============================================================
echo Weather Radar Prediction System - Quick Start
echo ============================================================
echo.

if "%WEATHER_RADAR_VIEWER_PORT%"=="" set WEATHER_RADAR_VIEWER_PORT=5050

echo Starting web viewer on http://localhost:%WEATHER_RADAR_VIEWER_PORT%
echo Open your browser and visit that URL
echo.
echo Press Ctrl+C to stop
echo ============================================================
echo.

python web_viewer.py
