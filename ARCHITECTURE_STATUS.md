# KV260 deployment status (verified 2026-11-18)

## Active software pipelines

```text
Browser photo upload
        │
        ▼
OpenCV EcoSort YOLOv8 30-class detector + waste classifier fallback (Arm CPU / PS)
        │ labels + short observation text
        ▼
Curated waste knowledge base + Llama 3.2 1B PTE using ExecuTorch (Arm CPU / PS)
        │
        ├── Piper English speech output (board)
        └── Vosk small English speech recognition (board; browser microphone input)
```

The OpenCV pipelines and Llama are separate. The vision model does not send image pixels to Llama. The board-side web service is at `/home/ubuntu/ecosort`, port 8000.

## Camera

The service scans `/dev/video*`, but the KV260 currently has none. `lsusb` shows only the onboard Microchip USB hubs, and `lsusb -t` shows no UVC camera. Linux has no `/dev/video*` or `/dev/media*`. The attached camera therefore is not enumerated by the OS. The board can accept uploaded photos, and the browser's local camera picker remains available. Software changes cannot make a physically absent USB device enumerate.

## Programmable logic (PL) accelerator

The current CV model executes through OpenCV DNN on the Arm CPU. A board-side `xbutil examine` reports **0 XRT devices found**. There is no `.xclbin`, `.bit`, or `.xmodel` in the board's project/runtime locations, and no Vitis AI/VART/XIR libraries were found. The Linux image has XRT 2.18, but XRT alone does not provide a configured DPU. The `uio` and FPGA device nodes are root-owned; the SSH user does not have passwordless sudo.

KV260 hardware supports DPUCZDX8G, but PL inference requires a matching loaded DPU design/bitstream, compatible runtime/driver and an architecture-compiled model. AMD's Vitis AI compiler produces a DPU-targeted XIR `.xmodel`; VART then executes it on the configured DPU. See [AMD Vitis AI: compiling for DPU](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Compiling-for-DPU) and [Vitis AI Runtime](https://docs.amd.com/r/2.0-English/ug1414-vitis-ai/Vitis-AI-Runtime). Therefore **this deployment has not moved OpenCV into PL**. Llama + ExecuTorch is on PS CPU as required.

## Model and observed limitations

- The active detector is `models/ecosort_yolov8n_30class_v1.onnx` (OpenCV DNN, Ultralytics output) with `models/ecosort_yolov8n_30class_v1_labels.txt`; the running process environment was verified after restart. It includes specific waste and e-waste labels.
- On the supplied Coke-can image, this 30-class model predicted `Earphone` at 56% and mapped it to e-waste. This is a confirmed failure case; evaluate against a larger labeled set before relying on predictions. The former YOLOX detector remains available for comparison only.
- The waste-material classifier remains configured as a fallback for detections without a mapped waste category. Its single-image accuracy is not established here.
- The active detector benchmark on the KV260 CPU (OpenCV DNN, four threads, input 640; 3 warmups, 15 runs) measured median 838.69 ms detection+postprocessing and 843.57 ms for the full `vision.describe` pipeline. This is a single-image latency measurement, not a dataset accuracy or sustained-load test.
- A focused regression with injected COCO labels verified that cell phone, laptop, keyboard, and mouse map to e-waste guidance without errors. This checks mapping logic, not object-detection accuracy.
- ETID and RealWaste dataset-preparation code is present. Neither dataset has been downloaded or used to train project-specific weights. The available Windows PyTorch DLL is blocked by Windows Application Control. See `data/README.md`.
- The deployed Llama 3.2 1B ExecuTorch PTE has a 128-token context. Prompts are kept brief; an incomplete or refusal output falls back to the curated KB. A fresh uncached e-waste advice request completed on board as `Recycle broken items to a designated E-waste centre.`

## Speech

KV260 `/api/health` reports `speech_input:true` and `speech_output:true`. Piper and Vosk are installed from local ARM64 wheels and local model files. A Piper-to-Vosk speech loopback returned text through `/api/transcribe`; the short synthesized test was imperfect (`where can i recycle and old mobile phone backslash and`), so speech recognition is operational but not verified for high accuracy. On desktop browsers, microphone capture requires a secure origin; use an SSH tunnel and open `http://localhost:8000`.

