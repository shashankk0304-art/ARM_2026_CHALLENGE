# Third-party notices

## EcoSort 30-class YOLOv8 detector

`models/ecosort_yolov8n_30class_v1.onnx` metadata declares an Ultralytics AGPL-3.0 license. The project owner confirms it was trained using the matching Roboflow Universe [Waste Detection Dataset 1](https://universe.roboflow.com/waste-detection-exuar/waste-detection-dataset-1), whose listing states CC BY 4.0. Attribute the dataset to Waste Detection and retain the license link. The ONNX has an additional `plastic straw` class that is not on the published 29-class page; its source and redistribution rights remain unverified. See `MODEL_PROVENANCE.md`.

## OpenCV Zoo YOLOX model

`models/object_detection_yolox_2022nov_int8bq.onnx` is the OpenCV Zoo YOLOX
COCO model, distributed under Apache License 2.0. Model source and license:
[OpenCV Zoo YOLOX](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox).
It recognizes the model's COCO classes; it is a general object detector. Detected
COCO object names such as phones and laptops are mapped to e-waste guidance.

## iAny waste-material classifier

`models/waste.onnx` and `models/labels.txt` are from [sengtha/iany-waste-v1](https://huggingface.co/sengtha/iany-waste-v1), released CC BY 4.0. It is an upstream MobileNetV2 classifier for can, glass, organic, other, paper, plastic bottle, and other plastic. Cite iAny/E-KHMER Technology and the upstream dataset terms. Its model card describes it as a suggestion, notes it has no e-waste class, and warns accuracy can drop on cluttered scenes. It is used with the OpenCV detector on the Arm CPU; it is not a model trained by this project.

## WasteWise classifier fallback

`models/wastewise_8class.onnx` and `models/wastewise_8class_labels.txt` are from [SriramRokkam/wastewise-garbage-cls](https://huggingface.co/SriramRokkam/wastewise-garbage-cls), licensed Apache-2.0. The model card describes an eight-class YOLOv8n classifier (battery, biological, cardboard, glass, metal, paper, plastic, trash). Its dataset and validation figures are upstream claims, not measurements made by EcoSort.

## Optional Piper LibriTTS-R voice

`setup_speech.sh` installs the bundled `en_US-libritts_r-medium` Piper ONNX voice and
its `MODEL_CARD`. Piper TTS 1.3.0 is GPL-3.0-or-later. The voice uses LibriTTS-R data under CC BY 4.0. Retain the
model card and attribution: [Piper voice model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/libritts_r/medium/MODEL_CARD).

## Optional Vosk offline speech recognition

`setup_speech.sh` installs Vosk 0.3.45 and `vosk-model-small-en-us-0.15`. The English model is intended for small devices and is Apache 2.0: [Vosk model list](https://alphacephei.com/vosk/models).

The same setup script also installs `vosk-model-small-hi-0.22` for Hindi offline input. The official Vosk model list identifies it as Apache 2.0; retain this notice and the upstream [model listing](https://alphacephei.com/vosk/models).

## Optional TACO training source

`prepare_taco_crops.py` consumes the public [TACO dataset](https://github.com/pedropro/TACO) annotations and image files. Each source image has its own recorded license; the converter keeps only entries with explicit CC0, CC BY, or CC BY-SA licensing and writes source attribution to `ATTRIBUTION.csv`. It excludes entries with missing/unclear or NC/ND license metadata. Include the dataset citation listed in the TACO README when reporting results.

## Public classifier training sources

`prepare_public_datasets.py` is intended for E-Tadweer and UCI RealWaste. Cite [ETID](https://data.mendeley.com/datasets/bndsgcspbx/2), DOI 10.17632/bndsgcspbx.2, and [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste). Preserve their CC BY 4.0 attribution when using or redistributing derived artifacts.
