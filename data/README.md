# Waste and e-waste data/model notes

The deployed site uses the bundled 30-class YOLOv8 ONNX detector for object detection, the iAny MobileNetV2 ONNX classifier for material classification, and the WasteWise ONNX classifier as a fallback. OpenCV DNN runs these models on the KV260 Arm CPU. The older OpenCV Zoo YOLOX model and backend are retained for comparison/benchmarking; they are not the active website detector. Llama 3.2 runs separately through ExecuTorch on the CPU.

The iAny classifier and WasteWise fallback are upstream pretrained weights; neither was trained by this project. A crop of the provided Coke-can image was classified as `can`/metal with 0.98 confidence by the board pipeline and as `metal` with 0.823 confidence by WasteWise. This is a single smoke check, not an accuracy estimate. The 30-class detector also has a documented can/earphone confusion. No project-specific training run or representative held-out evaluation was completed.

The source, training data and redistribution license for `models/ecosort_yolov8n_30class_v1.onnx` were not recorded in the files supplied for this handoff. Confirm and add those details before publicly redistributing that model. See [`../MODEL_PROVENANCE.md`](../MODEL_PROVENANCE.md) and [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Recommended public datasets

- **E-Tadweer (ETID-v2):** the dataset review lists 14,055 RGB e-waste images in 23 object classes, including phones, laptops, batteries, cables, keyboards, mice and PCBs. Mendeley lists the dataset under CC BY 4.0. Cite the published dataset: [Mendeley ETID](https://data.mendeley.com/datasets/bndsgcspbx/2), DOI `10.17632/bndsgcspbx.2`.
- **RealWaste:** 4,752 real-landfill images across nine household material categories. UCI lists CC BY 4.0: [UCI RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste).
- **TACO:** images in real settings with annotations for cans, bottles and other litter. Its image licenses are recorded per source photo. `prepare_taco_crops.py` keeps only entries with explicit CC0/CC BY/CC BY-SA licenses and writes attribution: [official TACO repository](https://github.com/pedropro/TACO).

## Prepare and train the classifier

Download the ETID-v2 and RealWaste archives from their linked dataset pages, then run from the project directory:

```powershell
python .\prepare_public_datasets.py --etid-zip .\data\ETID-v2.zip --realwaste-zip .\data\RealWaste.zip
python .\train_classifier.py .\data\public_waste_training 10
```

The preparation script copies only labeled image files, preserves source identity in filenames and writes dataset citations. Training reports held-out accuracy and per-class recall; do not claim those metrics unless training finishes and you inspect the output. It exports `models/waste.onnx` and `models/labels.txt`. OpenCV uses that classifier on appropriate detector crops and, when the detector misses an item, as a whole-image fallback. This is classification and does not retrain the active 30-class object detector.

Training requires PyTorch and torchvision. The available Windows PyTorch DLL is blocked by Application Control, so no project-specific classifier weights have been trained. The bundled upstream classifier is deployed as a baseline, but ETID/RealWaste archives have not been merged into new weights. Train on a compatible machine and copy only validated weights and labels to the KV260. Dataset accuracy alone does not guarantee accuracy for your webcam; keep your own varied camera photos in a separate held-out test set.

