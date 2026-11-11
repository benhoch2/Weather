# Rain Nowcast

Short-term rain forecasts (0–60 minutes) for Israel from the Israel Meteorological Service (IMS) rain radar,
using a U-Net trained on the GPU and scored against simple baselines.

## How it works

1. **Data.** IMS publishes a transparent 940×940 rain-only radar image every 5 minutes, and keeps an archive
   back to at least January 2025. Each pixel's color is converted to a rain rate in mm/h using the IMS legend.
   Frames are stored by their real observation time (UTC).
2. **Model.** A U-Net takes the last 12 frames (60 minutes) at 256×256 and predicts the next 12 frames
   (+5 … +60 minutes) in one pass. The loss weights heavier rain more, so "no rain everywhere" can't score well.
3. **Scoring.** Each forecast is checked against what actually happened, and so are two baselines:
   *persistence* (rain stays where it is) and *optical flow* (rain keeps moving along its current track,
   computed with pysteps). Scores are CSI, POD and FAR at 0.5, 2 and 5 mm/h, plus MAE in mm/h.
   CSI is the share of rain areas forecast correctly, counting misses and false alarms; 1 is perfect.
4. **Learning.** The live system never changes model weights. Once a night in rainy weather, `retrain.py`
   fine-tunes on all data. The new model replaces the live one only if it scores better on held-out days.

## Setup (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
```

RTX 50-series (Blackwell) cards need a CUDA 12.8+ build of PyTorch, as above. TensorFlow has no GPU
support on native Windows. Without a GPU everything still runs on the CPU, but training is very slow.

## First run

```powershell
# 1. Download history: hourly scan since Jan 2025, then every 5-minute frame around rainy hours.
#    Takes several hours; it is throttled and resumable (just run it again).
.\.venv\Scripts\python.exe backfill.py

# 2. Score the baselines on held-out days. This is the bar the model must beat.
.\.venv\Scripts\python.exe evaluate.py --split test --methods persistence,optflow

# 3. Train on the GPU (check the pipeline first with --smoke), then compare on the test days.
.\.venv\Scripts\python.exe train.py --smoke
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py --split test

# 4. Run everything: fetcher + live forecasts + web viewer at http://localhost:5050
start_viewer.bat
```

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, timing, model size, IMS URLs, legend and map bounds |
| `ims_source.py` | IMS listing, archive URLs, downloads, color → mm/h decoding and rendering |
| `radar_store.py` | Raw archive in `data/ims/raw/`, gap tracking, per-day 256×256 cache (`python radar_store.py` builds it in parallel) |
| `backfill.py` | Historical downloader (`--from/--to`, `--all`) |
| `fetcher.py` | Live polling, plus repair of any gaps in the last 48 hours |
| `dataset.py` | Time-ordered frames, gap-free windows, train/val/test split by whole days |
| `model.py` / `train.py` | U-Net, checkpoints, GPU training, promote-if-better |
| `metrics.py` / `baselines.py` / `evaluate.py` | Scores, persistence and optical-flow baselines, offline reports in `data/eval/` |
| `predictor.py` | Live forecasts in `data/predictions/<t0>/`, scoring, and `ledger.jsonl` |
| `retrain.py` | Nightly gated fine-tuning (`--force` to run now) |
| `web_viewer.py` | Map with observed and forecast frames, accuracy charts, and system status |
| `start.py` | Supervisor: restarts crashed parts, honors the viewer's pause button, runs the nightly retrain |

## Notes

- The viewer listens only on `127.0.0.1`. Use `start.py --host 0.0.0.0` to open it to your network; it has no login.
- The IMS archive has outages (missing frames). Training only uses gap-free windows. The live predictor
  tolerates one missing input frame.
- Radar data © Israel Meteorological Service. Check the IMS terms of use before redistributing data or forecasts.
- Run the tests with `.\.venv\Scripts\python.exe -m pytest tests`.
