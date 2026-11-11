# Publishing at rain.musktzur.uk

The app runs on this PC (it has the GPU). benh-pi's existing Cloudflare Tunnel gets one more
ingress rule that forwards `rain.musktzur.uk` to the PC over Tailscale. Cloudflare Access puts a
login in front of it.

```
browser → Cloudflare edge (Access login) → tunnel "benh-pi" on benh-pi
        → Tailscale → this PC 100.79.167.68:5050 (waitress, --require-access)
```

Nothing changes on the TP-Link router. Nothing on the home Wi-Fi can reach the app:
it listens only on localhost and the PC's Tailscale address, and the firewall rule
admits only benh-pi.

## Go-live steps, in this order

The order matters. Access is set up before DNS exists, so the hostname is never public without a login.

1. **Cloudflare Access** (dashboard, done by Ben): Zero Trust → Access → Applications →
   *Add* → Self-hosted. Domain `rain.musktzur.uk`. Add a policy *Allow* → *Emails* → the
   addresses that may log in. Login method: one-time PIN.
2. **Windows firewall** (admin PowerShell on this PC), allowing benh-pi only:
   ```powershell
   New-NetFirewallRule -DisplayName "Rain nowcast from benh-pi (Tailscale)" -Direction Inbound `
     -Protocol TCP -LocalPort 5050 -RemoteAddress 100.100.146.14 -Action Allow -Profile Any
   ```
3. **Start the app**: `publish.bat`. It listens on `127.0.0.1` and `100.79.167.68`, port 5050,
   with `--require-access`. Then check from benh-pi:
   `curl -s -o /dev/null -w '%{http_code}\n' http://100.79.167.68:5050/` should print `403`.
   That's the expected result: the request has no Access token.
4. **Tunnel ingress on benh-pi**: add the rule *above* the catch-all in `/etc/cloudflared/config.yml`:
   ```yaml
     - hostname: rain.musktzur.uk
       service: http://100.79.167.68:5050
   ```
   Then (from the BeerServer runbook; never SIGHUP):
   ```bash
   sudo systemd-run --on-active=15min --unit=cf-deadman systemctl stop cloudflared   # safety net
   sudo cloudflared tunnel ingress validate
   sudo systemctl restart cloudflared
   cloudflared tunnel route dns --overwrite-dns benh-pi rain.musktzur.uk
   # verify, then cancel the dead-man switch:
   sudo systemctl stop cf-deadman.timer
   ```
5. **Verify**: open https://rain.musktzur.uk in a private window. You should get the Cloudflare
   Access login, then the app after logging in. `dig rain.musktzur.uk` should show only Cloudflare
   addresses, never our IP.

## Notes

- **The site is down while this PC is off or asleep.** Sleep on AC power is set to never. Closing a
  laptop lid may still sleep it (Control Panel → Power Options → *Choose what closing the lid does*).
- **Roll back**: remove the ingress rule and restart cloudflared, or just stop `publish.bat`.
  Access keeps the hostname protected either way.
- **Data credit**: radar data © Israel Meteorological Service (credited in the page footer). Check the
  IMS terms of use before sharing beyond a private group.
- The `--require-access` check only looks for the `Cf-Access-Jwt-Assertion` header; it doesn't verify
  the signature. It's a safety net in case the Access policy is removed. The real gate is Access itself.
