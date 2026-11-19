# EcoSort: On-device AI Waste Assistant — Technical Report

## 1. Objective
EcoSort helps users identify waste, get disposal guidance, learn recycling options and environmental facts, using Llama on ExecuTorch on an Arm device.

## 2. Requirement traceability
| Requirement | Implementation/status |
|---|---|
| Classify waste | OpenCV DNN with the EcoSort 30-class YOLOv8 ONNX detector, plus the waste-material classifier fallback; CPU/PS. Best-effort automatic classification; this run did not train new weights. |
| Disposal recommendations and recycling | Curated knowledge base and optional Llama advice through `/api/advise`. |
| Environmental education | Knowledge-base facts and Q&A through `/api/ask`. |
| Llama and ExecuTorch | Llama 3.2 1B PTE via installed ExecuTorch `llama_main` on the Arm CPU/PS. |
| User interface | KV260-hosted web app with automatic photo analysis, multilingual chat/Q&A and browser-camera live detection. |
| Speech | Piper English TTS and Vosk English ASR on KV260. The packaged source now configures Vosk English/Hindi input; Hindi setup and board inference have not yet been verified on the KV260. Telugu/Kannada/Marathi recognition and non-English TTS still depend on browser services/installed voices. Browser microphone/camera capture needs localhost/HTTPS. |

## 3. Architecture
```text
photo / browser or USB camera -> OpenCV YOLOv8 (30 classes) + material classifier (CPU/PS)
                    -> detected labels + short text -> knowledge base / Llama 3.2 1B via ExecuTorch (CPU/PS)
                    -> disposal guidance -> Piper TTS (board)

browser camera -> compressed frame uploads -> OpenCV detector/classifier (board)
browser microphone -> Vosk ASR (board; English/Hindi models after running setup_speech.sh)
```
Vision and language are separate inference stages. The CV pipeline sends labels and short text, not image pixels, to the language model. No cloud inference is used.

## 4. Hardware and deployment
The target is a Kria KV260 running Ubuntu 24.04.2, aarch64. The app is deployed at `/home/ubuntu/ecosort` and serves port 8000. The current service is reachable at `http://192.168.137.2:8000` on the user's LAN. The deployed model and runner are detected by `/api/health`.

## 5. Language model
The board has an existing Llama 3.2 1B PTE (802,018,816 bytes), tokenizer, and ExecuTorch runner. Its exported context is 128 tokens; `LLAMA_MAX_NEW=40` limits generation requests but does not enlarge that context. Chat responses now show **Advice**, **Reasoning**, and **Learn**: Llama generates one short action sentence, then the app adds grounded impact/recycling facts from the KB. This avoids dropping educational content when Llama cannot fit three complete sections in its context. A 2048-token replacement PTE has not been exported or validated on the board.

## 6. Vision model and data
The active object detector is `models/ecosort_yolov8n_30class_v1.onnx`, with labels in `models/ecosort_yolov8n_30class_v1_labels.txt`. Its 30 classes include bottles, cans, food/paper/plastic groups and e-waste objects such as phones, chargers, cables, bulbs and batteries. The former OpenCV Zoo YOLOX detector is retained only for comparison. The material classifier serves as a fallback where detection does not produce a mapped category. The custom detector's single Coke-can check was incorrect, so its class labels still require evaluation on a representative labeled set.

The interface automatically runs material classification when detection does not supply a mapped waste category; users do not need to choose a material for recognized items. A crop of the Coke can shown in the user's screenshot was uploaded through the board API and classified as `can`/metal with 0.98 confidence. This is one smoke check and is not an accuracy estimate.

The project includes preparation scripts for E-Tadweer ETID-v2 and UCI RealWaste, plus TACO license-filtered crop preparation. Public sources: [ETID-v2 (CC BY 4.0)](https://data.mendeley.com/datasets/bndsgcspbx/2), [RealWaste (CC BY 4.0)](https://archive.ics.uci.edu/dataset/908/realwaste), and [TACO](https://github.com/pedropro/TACO). ETID-v2 includes e-waste categories; RealWaste expands household material coverage. The archives were not downloaded/merged in this run and no project-specific weights were trained. Windows Application Control blocked the available PyTorch DLL, so there are no dataset-wide validation metrics. The deployed classifier is an upstream pretrained baseline, not a model trained here. Refer to `data/README.md` and `THIRD_PARTY_NOTICES.md` for commands, limits, and attribution.

## 7. Camera and FPGA status
At the time of board inspection there was no `/dev/video*` or `/dev/media*` device. `lsusb` showed only internal Microchip hubs; the external webcam was not enumerated. The site now offers a browser-camera live mode: it captures frames on the user's computer/phone, sends them to `/api/capture`, and the KV260 performs the detection. The physical USB camera attached to the board still requires enumeration before board-local capture can work. For browser camera or microphone permissions, create an SSH local forward on port 8000 and open `http://localhost:8000`.

`xbutil examine` reported zero XRT devices. No configured DPU bitstream/xclbin, compiled XIR `.xmodel`, or Vitis AI runtime was found in the inspected board locations. Thus OpenCV CV and ExecuTorch Llama both run on Arm CPU/PS. To run CV on PL, the board needs a matching DPU design/bitstream, runtime/driver and a model compiled for that DPU. See [AMD DPU compilation](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Compiling-for-DPU) and [Vitis AI Runtime](https://docs.amd.com/r/2.0-English/ug1414-vitis-ai/Vitis-AI-Runtime).

## 8. Observed board checks
| Check | Result |
|---|---|
| `/api/health` | Detector, material classifier, Llama model/runner/tokenizer, speech input and output available; camera unavailable. |
| Coke-can image crop | Board `/api/capture`: `can`/metal, confidence 0.98 (single smoke test). |
| Llama/ExecuTorch e-waste advice | Completed with `Recycle broken items to a designated E-waste centre.` |
| Llama/ExecuTorch can advice | Completed with `Aluminium cans can be recycled and reused as metal.` |
| COCO e-waste label mapping regression | Injected detector labels `cell phone`, `laptop`, `keyboard`, and `mouse` map to e-waste without errors. This verifies label handling, not real detector accuracy. |
| Piper speech output | `/api/speak` returned HTTP 200 and a valid 22.05 kHz mono WAV (104,492 bytes). |
| Piper → Vosk loopback | Speech endpoints completed; synthetic phrase transcription was imperfect. |
| Active 30-class YOLOv8 CPU latency | On the KV260, OpenCV-DNN with 4 OpenCV threads, 640 detector input, 3 warmups and 15 runs: median detector+postprocess 838.69 ms (mean 838.61 ms; 1.19 serial frames/s). Full `vision.describe` median 843.57 ms (10 runs). The running website process was verified with `YOLO_FORMAT=ultralytics`, the 30-class ONNX path, its labels path, and input size 640. |
| 30-class can check | On the provided Coke-can image, active model + fallback returned `Earphone`/e-waste at 56%. This is a demonstrated failure case, not an accuracy estimate. |
| Former YOLOX baseline CPU latency | Before the model switch, OpenCV-DNN with 4 threads, 640 input, 5 warmups and 30 runs: median detector+postprocess 2390.55 ms (mean 2391.57 ms; 0.42 serial frames/s); full `vision.describe` median 2488.30 ms (10 runs). The baseline is no longer active. |
| Multilingual UI/API | Redesigned UI options and chat language parameters are present for English, Hindi, Telugu, Kannada and Marathi. The current source adds offline Hindi Vosk selection and clearer browser camera/recognition errors; this source update has not yet been installed or tested on the KV260. Telugu/Kannada/Marathi input still uses browser speech recognition (may require internet). English output uses Piper; other languages need an installed matching browser voice. |
| Structured chatbot | A live English battery Q&A returned `src=llama-grounded`: Llama generated an Advice sentence and the app added curated battery hazard and metal-recovery facts as Reasoning and Learn. Response time was 42.5 s on this board run. |
| Hindi chatbot check | A Hindi battery question mapped to the battery category and returned localized Advice plus KB Reasoning/Learn (`src=kb`) after 48.3 s; the short-context model did not produce a complete advice sentence. Hindi Llama generation remains unverified. |
| Board `/api/health` at last inspection | Detector, classifier, Llama model/runner/tokenizer, English speech input and Piper output report ready; board USB camera reports unavailable. Hindi Vosk support is in the submission source but not yet installed on the board. |
| Latency outside the listed image benchmark, sustained throughput, peak RAM, dataset accuracy | Not measured. Do not infer these from the smoke checks. |

## 9. Limitations and next steps
The active 30-class detector has a demonstrated can/earphone confusion; evaluate it on a representative labeled test set before relying on its predictions. Train and evaluate on legally usable, correctly labeled ETID-v2 and RealWaste data plus a held-out set of local camera images. Keep dataset sources and class mapping explicit. Laptop live-camera use requires browser camera permission from localhost/HTTPS; the board USB camera remains unavailable. PL acceleration remains future work until a compatible DPU image/runtime/model are provided and loaded. Hindi on-board ASR still needs installation and testing using `setup_speech.sh`; Telugu/Kannada/Marathi ASR and non-English TTS depend on browser services/voices. Recycling rules vary by municipality, so users should check local collection guidance.
