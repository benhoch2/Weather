# Running on benh-pi and publishing at rain.musktzur.uk

**Split (option A):**

| benh-pi (always on) | This PC (GPU, when it's on) |
|---|---|
| `fetcher.py`: live IMS frames | `backfill.py`: the history download |
| `predictor.py`: forecasts every 5 min with the small model (ONNX via OpenCV, no PyTorch) | `train.py` / `retrain.py`: training, promoted only if better |
| `web_viewer.py`: the site, on `127.0.0.1:5050` | `push_model.py`: copies a promoted model to the Pi |

```
browser → Cloudflare edge (Access login) → tunnel "benh-pi" → 127.0.0.1:5050 on benh-pi
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
it up within 30 seconds. The service keeps forecasts for 7 days (SD card), caps itself at 450 MB of RAM, and runs
at lower priority than the other apps.

Code updates use the `deploy` skill (`.claude/deploy.json`): `git pull` + `systemctl restart rain`.

## Publishing, in this order

The order matters. Access is set up before DNS exists, so the hostname is never public without a login.

1. **Cloudflare Access** (dashboard, done by Ben): Zero Trust → Access → Applications → *Add* → Self-hosted.
   Domain `rain.musktzur.uk`. Policy *Allow* → *Emails* → the people who may log in. One-time PIN login.
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
3. **Verify**: https://rain.musktzur.uk in a private window should show the Access login, then the app.
   `dig rain.musktzur.uk` should show only Cloudflare addresses.

## Safety net

The service runs with `--require-access`. Any request that came through Cloudflare (it carries `Cf-Connecting-IP`)
but has no `Cf-Access-Jwt-Assertion` header gets a 403. If the Access policy is ever removed, the site
refuses requests instead of going public. The check doesn't verify the token's signature; Access itself is the real gate.

## Notes

- Radar data © Israel Meteorological Service (credited in the page footer). Check the IMS terms before sharing widely.
- Baseline scoring on the Pi uses OpenCV optical flow (pysteps isn't installed there), so its optical-flow numbers
  can differ slightly from the PC's offline evaluation.
