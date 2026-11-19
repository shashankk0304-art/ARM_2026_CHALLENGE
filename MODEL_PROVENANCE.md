# Model provenance checklist

This file records which model artifacts are included and what is known about their origin. Keep the associated notices with any redistribution.

| Artifact | Use | Provenance / license status |
|---|---|---|
| `models/ecosort_yolov8n_30class_v1.onnx` and `models/ecosort_yolov8n_30class_v1_labels.txt` | Active 30-class OpenCV DNN detector | ONNX metadata declares `AGPL-3.0 License (https://ultralytics.com/license)` and describes it as an Ultralytics model. Metadata records the training config path `/content/Roboflow_Waste_v3_Balanced_v2/data.yaml`. The project owner confirms this model uses the matching Roboflow Universe project [Waste Detection Dataset 1](https://universe.roboflow.com/waste-detection-exuar/waste-detection-dataset-1), listed as CC BY 4.0; attribute the dataset to Waste Detection and retain the license link. The published page lists 29 classes, while this model has a 30th label (`plastic straw`). The source and redistribution rights for that additional class are not confirmed; do not claim the full model's training data is CC BY 4.0 until this is resolved. SHA-256: `1a1279d54ca288a0d32a6f0c83cea14c9ad0c5493bb599804c15decdc7bb9dcc`. |
| `models/waste.onnx`, `models/labels.txt` | iAny material classifier | Upstream iAny/E-KHMER Technology model, CC BY 4.0. See `THIRD_PARTY_NOTICES.md`. |
| `models/wastewise_8class.onnx`, `models/wastewise_8class_labels.txt` | Fallback classifier | Upstream WasteWise model, Apache-2.0. See `THIRD_PARTY_NOTICES.md`. |
| `models/object_detection_yolox_2022nov_int8bq.onnx` | Older comparison detector | OpenCV Zoo YOLOX, Apache-2.0. See `THIRD_PARTY_NOTICES.md`. |
| `wheelhouse/speech/en_US-libritts_r-medium.onnx` | English Piper speech | LibriTTS-R voice, CC BY 4.0; model card is included alongside it. |
| `wheelhouse/speech/vosk-model-small-en-us-0.15.zip` | Offline English speech recognition | Vosk model, Apache-2.0. |

The Llama 3.2 1B ExecuTorch `.pte`, tokenizer and `llama_main` executable are installed separately on the KV260 and are intentionally not copied into this source folder. The README documents their expected paths and context limit. Follow the Llama license and model access terms when distributing those files.
