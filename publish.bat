@echo off
rem Runs the system for rain.musktzur.uk: the viewer listens on localhost and on this
rem PC's Tailscale address (where benh-pi's Cloudflare tunnel connects), and refuses
rem any remote request that didn't come through Cloudflare Access. See PUBLISHING.md.
cd /d %~dp0
if "%TAILSCALE_IP%"=="" set TAILSCALE_IP=100.79.167.68
".venv\Scripts\python.exe" start.py --host 127.0.0.1,%TAILSCALE_IP% --require-access %*
