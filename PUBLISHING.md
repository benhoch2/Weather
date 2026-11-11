# Running on benh-pi and publishing at rain.musktzur.uk

**Split (option A):**

| benh-pi (always on) | This PC (GPU, when it's on) |
|---|---|
| `fetcher.py`: live IMS frames | `backfill.py`: the history download |
| `predictor.py`: forecasts every 5 min with the small model (ONNX via OpenCV, no PyTorch) | `train.py` / `retrain.py`: training, promoted only if better |
| `web_viewer.py`: the site, on `127.0.0.1:5050` | `push_model.py`: copies a promoted model to the Pi |

```
browser → Cloudflare edge → tunnel "benh-pi" → 127.0.0.1:5050 on benh-pi (password page)
```

The Pi runs the small `--norm batch` model (1.9M parameters) exported to ONNX. PyTorch doesn't exist
for the Pi's 32-bit OS, and OpenCV's DNN module runs the model instead.

## One-time Pi setup

```bash
sudo apt install -y python3-opencv python3-waitress        # numpy, pillow, flask, requests, psutil already there
git clone https://github.com/benhoch2/Weather.git ~/Weather && cd ~/Weather
git checkout rebuild-ims-pytorch                             # until it's merged
sudo cp deploy/rain.service /etc/systemd/system/rain.service
sudo systemctl daemon-reload && sudo systemctl enable --now rain
```

From the PC: `.venv\Scripts\python.exe push_model.py` copies the current Pi model over. The Pi's predictor picks
it up within 30 seconds. The service keeps forecasts for 7 days (SD card), uses about 255 MB of RAM, and runs
at lower priority than the other apps.

Code updates use the `deploy` skill (`.claude/deploy.json`): `git pull` + `systemctl restart rain`.

## Password

The site uses one shared password. Set it on the Pi, in your own terminal (it's never echoed, logged or
stored in plain text; only a salted hash goes to `~/Weather/data/auth/password.json`):

```bash
ssh benh@benh-pi
cd ~/Weather && python3 auth.py set-password
```

Each browser asks once, then a signed cookie keeps it logged in for about 400 days. Running
`set-password` again changes the password and logs everyone out. After 5 wrong attempts, an address
has to wait 15 minutes. Until a password is set, remote visitors get "not configured" (fails closed).
Requests made on the Pi itself, not through the tunnel, need no password.

## Publishing (tunnel)

1. Set the password (above).
2. **Tunnel ingress on benh-pi**: add the rule *above* the catch-all in `/etc/cloudflared/config.yml`:
   ```yaml
     - hostname: rain.musktzur.uk
       service: http://localhost:5050
   ```
   ```bash
   sudo systemd-run --on-active=15min --unit=cf-deadman systemctl stop cloudflared   # safety net
   sudo cloudflared tunnel ingress validate
   sudo systemctl restart cloudflared                                                  # never SIGHUP
   cloudflared tunnel route dns --overwrite-dns benh-pi rain.musktzur.uk
   sudo systemctl stop cf-deadman.timer                                                # once verified
   ```
3. **Verify**: https://rain.musktzur.uk should show the password page. `dig rain.musktzur.uk` should show
   only Cloudflare addresses.

## Notes

- Radar data © Israel Meteorological Service (credited in the page footer). Check the IMS terms before sharing widely.
- Baseline scoring on the Pi uses OpenCV optical flow (pysteps isn't installed there), so its optical-flow numbers
  can differ slightly from the PC's offline evaluation.
