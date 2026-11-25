# Run EcoSort on a fresh Kria KV260

These steps assume the target is an Arm64 KV260 running Ubuntu 24.04 with Python 3.12. The repository contains the app, active CV models, and offline Arm64 wheels. The ExecuTorch runner and Llama model files are separate prerequisites and are not in Git.

## 1. Clone the repository on the board

Replace `<GIT_URL>` with the URL of the project repository:

```bash
cd "$HOME"
git clone <GIT_URL> ecosort
cd "$HOME/ecosort"
uname -m
python3 --version
```

Expected platform output is `aarch64` and Python 3.12.x. If virtual-environment support is missing, install it before continuing:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

## 2. Install the bundled application and speech dependencies

```bash
cd "$HOME/ecosort"
bash setup_kv260.sh
bash setup_speech.sh
```

These use the bundled Arm64 wheels without pip downloading packages. `setup_speech.sh` installs Piper English TTS and Vosk English/Hindi ASR assets. Telugu, Kannada, and Marathi input still use the browser speech service; non-English TTS requires a matching voice on the browser device.

## 3. Install/provide ExecuTorch Llama files

The app requires a working Arm64 ExecuTorch `llama_main` executable, a compatible Llama 3.2 1B XNNPACK `.pte`, and the matching tokenizer. They are intentionally not bundled in Git. Install or copy these files onto the board, then set the paths in the same shell used to start EcoSort:

```bash
export LLAMA_RUNNER="$HOME/executorch/cmake-out/examples/models/llama/llama_main"
export LLAMA_MODEL="$HOME/models/llama32_1b_xnnpack.pte"
export LLAMA_TOKENIZER="$HOME/models/tokenizer.model"
```

If your files are elsewhere, replace those three values with their actual paths. The model context is 128 tokens in the current board deployment; keep `LLAMA_MAX_CTX` at or below the exported model context.

## 4. Check required files and start

```bash
cd "$HOME/ecosort"
bash check_runtime.sh
bash run.sh
```

The preflight checks Python dependencies, the 30-class YOLO model and labels, the classifier assets, speech assets, the ExecuTorch runner/model/tokenizer, and whether port 8000 is free. If it reports that port 8000 is occupied, test the existing app instead of starting a duplicate:

```bash
curl http://127.0.0.1:8000/api/health
```

To keep the server running after the SSH session closes, start it once in the background instead of running the foreground command:

```bash
nohup bash run.sh >>app.log 2>&1 </dev/null &
```

Watch logs with `tail -f app.log`. Don’t launch a second copy while port 8000 is occupied.

## 5. Open the site from your laptop — use this one browser address

The EcoSort server runs on the KV260 at port `8000`. From a laptop, use an SSH tunnel so the browser gets a secure localhost origin for camera and microphone APIs. Start this in Windows PowerShell (replace `<KV260_IP>` with the board's current address):

```powershell
ssh -N -L 8001:127.0.0.1:8000 ubuntu@<KV260_IP>
```

Keep the PowerShell tunnel window open. In the laptop browser, use **only** `http://localhost:8001/` for the project; this same page handles photo upload, laptop live camera, and microphone input. Allow camera and microphone access when the browser asks. Do not browse to `http://<KV260_IP>:8000/` on the laptop when using its camera or microphone: a plain-HTTP IP address is not a secure browser origin. Port `8000` is the KV260-side service port forwarded internally by the tunnel; it is not a second website to use.

If SSH reports `channel ... open failed: connect failed: Connection refused`, the tunnel reached the board but the EcoSort app is not listening on board port `8000`. In another PuTTY shell, start the app with `bash run.sh` from `$HOME/ecosort` and keep it running. To check the app on the board, run `curl http://127.0.0.1:8000/api/health`. From the laptop, check the same service through the tunnel at `http://localhost:8001/api/health`.

A health response with `camera:false` means no USB camera was found on the KV260; it does not report the laptop camera. Live laptop frames are captured by the browser and sent to the board for OpenCV detection.

The active detector is `models/ecosort_yolov8n_30class_v1.onnx`, selected by `run.sh` with `YOLO_FORMAT=ultralytics`. Its labels are in `models/ecosort_yolov8n_30class_v1_labels.txt`. The iAny material classifier (`models/waste.onnx`) and WasteWise fallback are separate classifier stages. Current inference runs on the Arm CPU/PS; PL acceleration is not configured.

## Git and model files

The `models/` folder is part of the repository and is not ignored. The Llama PTE, tokenizer, and `llama_main` remain external board prerequisites. The active 30-class ONNX metadata declares Ultralytics AGPL-3.0, but its exact Roboflow training export and the added `plastic straw` class provenance are not verified; review [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistribution. The demonstration video is a separate challenge deliverable; see [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).
