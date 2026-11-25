# EcoSort — on-device waste assistant

Photo or browser/USB camera → OpenCV DNN with EcoSort's 30-class YOLOv8 ONNX detector and waste-material classifier on the KV260 Arm CPU → category and curated disposal facts → Llama 3.2 via ExecuTorch on the same Arm CPU. The CV and Llama pipelines are local and separate; the CV pipeline sends only labels and short text to Llama. Piper TTS and Vosk ASR also run locally on the Arm CPU.

## KV260 setup and run

Follow the complete fresh-board sequence in [RUN_ON_KV260.md](RUN_ON_KV260.md). In short: clone the repository to the KV260, run `bash setup_kv260.sh` and `bash setup_speech.sh`, provide the separate ExecuTorch runner/PTE/tokenizer paths, then run `bash run.sh`. `check_runtime.sh` validates required files, dependencies, the active YOLO model, and port 8000 before the app starts.

The ExecuTorch `llama_main`, Llama 3.2 1B XNNPACK `.pte`, and tokenizer are not bundled in Git. A new board must have those files installed separately; the exact default paths and override instructions are in the run guide. For laptop use, open the site only at **`http://localhost:8001/`** after starting the SSH tunnel described in the guide. This one browser address supports photo upload, laptop camera, and microphone access. Port `8000` is the app's internal port on the KV260 and is the tunnel destination; do not open the KV260 IP URL in the laptop browser when camera or microphone access is needed.

The active detector is `models/ecosort_yolov8n_30class_v1.onnx` with the matching labels file. It includes specific e-waste and common waste classes. A 30-class check on the provided Coke-can image incorrectly predicted `Earphone` at 56%, so verify detections against real objects; the waste classifier remains as a fallback when the detector does not return a mapped category. The former OpenCV Zoo YOLOX detector is no longer the active website model. See [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The current KV260 board has no usable USB camera device: `/dev/video*` is absent and `lsusb` shows only the board's internal hubs. Phone/photo upload still works. Plugging the camera into the host port and seeing it enumerate as a UVC device must be checked physically.

**FPGA status:** This deployment currently runs OpenCV detection/classification and Llama on the Arm CPU (PS). `xbutil examine` reports zero XRT devices; no DPU bitstream/xclbin, compiled `.xmodel`, or Vitis AI runtime is present. Therefore CV is not running in PL yet. The KV260 can use DPUCZDX8G with a matching bitstream, runtime and architecture-compiled model, but those hardware/software components must be installed before that pipeline can be enabled.

## Important Llama context limit

The board's existing PTE has a 128-token context. The wrapper now passes `--max_new_tokens` to the installed runner, but the PTE still caps prompt plus response to its exported context. `LLAMA_MAX_NEW` cannot enlarge that. `export_llama.sh` defines an export from the matching Llama 3.2 1B Instruct checkpoint and params at 2048 context using 8da4w mixed quantization. Run it from an ExecuTorch source checkout on a capable Linux machine, validate the output with the target `llama_main`, and only then copy the PTE to the board. `LLAMA_MAX_CTX` must never exceed the PTE context.

The board runner/model/tokenizer defaults are set in `run.sh`. `LLAMA_MAX_NEW` requests more generation tokens but cannot exceed the PTE context. The UI always presents grounded knowledge-base disposal and recycling facts even when Llama reaches its output limit.

Chat answers are displayed as **Advice**, **Reasoning**, and **Learn**. Llama generates the concise advice sentence; the reasoning and recycling/environmental facts come from the curated knowledge base so they remain grounded when the short-context model cannot finish a longer response. `LLAMA_MAX_NEW` defaults to 40 for this 128-token PTE. This is a context workaround, not a context-window expansion.

## Voice

**Speak** is on by default. The board uses Piper TTS for English. `setup_speech.sh` installs English and Hindi Vosk ASR models for on-board, offline microphone recognition. Telugu, Kannada, and Marathi recognition use the browser's Web Speech service and may require Internet access. Board TTS is English-only; other-language speech output requires a matching voice installed on the browser device. On Windows, add the language and its Text-to-speech feature in **Settings → Time & language → Language & region**; availability varies by language/Windows edition. See Microsoft's [language and speech setup guide](https://support.microsoft.com/en-us/windows/hardware/input-devices/manage-the-language-and-keyboard-input-layout-settings-in-windows).

## Languages and live detection

The refreshed learning UI and chat support English, Hindi, Telugu, Kannada, and Marathi. The language selector localizes the main page controls, category/bin labels, Llama response instruction, and speech language. English and Hindi microphone input use offline Vosk on the KV260. Telugu, Kannada, and Marathi microphone input use the browser Web Speech service when available; it may need Internet and is not processed on the KV260. Answers use Piper on the board for English and a matching browser voice for other languages. Non-English speech reads the Llama advice line; the full grounded reasoning and learning facts remain visible in chat. Browser voice availability varies by device. Use HTTPS or the PuTTY localhost tunnel for mic/camera permissions.

**Use laptop camera (live)** captures the browser device's camera and sends one compressed frame at a time to the KV260 `/api/capture` endpoint. The board runs OpenCV detection/classification on its CPU and returns annotated frames. Use the PuTTY localhost SSH tunnel above (or HTTPS) so the browser allows camera access. The KV260 currently has no enumerated USB webcam, so the laptop camera is the live input. When no board camera is detected, the duplicate board-capture button is hidden; the remaining **Choose photo** button opens a file/photo picker.
## Files and validation

- `app.py`, `templates/index.html`: web UI and local API.
- `vision.py`: OpenCV YOLOv8 object detection plus the waste-material classifier; `yolox_backend.py` supports the older comparison model.
- `llm_executorch.py`: local ExecuTorch CLI wrapper.
- `waste_kb.py`, `labels_map.json`: disposal knowledge and label mapping.
- `prepare_public_datasets.py`, `train_classifier.py`: ETID/RealWaste dataset preparation and classifier training for a compatible training host.
- `requirements.txt`, `requirements_speech.txt`, `wheelhouse/`, `setup_kv260.sh`, `setup_speech.sh`: offline Arm64 install.
- `check_runtime.sh`, `RUN_ON_KV260.md`: required-file preflight and fresh-board run sequence.
- `REPORT.md`: technical report with verified smoke checks and explicitly unmeasured results.
- `MODEL_PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`: model/runtime attribution and redistribution checklist.
- `DEMO_SCRIPT.md`: outline for recording the required challenge demo video.

Check the live service at `/api/health`. A `camera:false` result is expected until a camera is connected; phone image upload remains available. Never report placeholder or unmeasured accuracy/latency as a result.

## Submission notes

The source folder contains the OpenCV ONNX models, Arm64 offline Python wheels, speech assets, UI, setup scripts, and reports. The Llama `.pte`, tokenizer, and ExecuTorch `llama_main` binary are not included; they are installed separately on the KV260 and are not tracked in this repository. The checked-in launch script documents their board paths. The active 30-class detector's ONNX metadata declares the Ultralytics AGPL-3.0 license and records a Roboflow training-config path, but the exact dataset version and rights for the extra `plastic straw` class are not confirmed. Review `MODEL_PROVENANCE.md` and `THIRD_PARTY_NOTICES.md` before redistributing the model. The required demo video is also not present yet; use `DEMO_SCRIPT.md` to record it.




## Final project report

- [PDF report](docs/EcoSort_Final_Project_Report.pdf)
- [Editable Word report](docs/EcoSort_Final_Project_Report.docx)

