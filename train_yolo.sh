#!/bin/bash
# PC (GPU recommended): pip install ultralytics. data.yaml = a YOLO-format waste dataset (e.g. TACO converted, or a public waste set; check its licence).
yolo detect train data=data.yaml model=yolov8n.pt imgsz=416 epochs=60 batch=16
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=416 opset=12 simplify=True
mkdir -p models && cp runs/detect/train/weights/best.onnx models/yolo.onnx
# Put the class names from data.yaml (one per line, same order) into models/yolo_labels.txt, then copy models/ to the board.
# On the board, lower YOLO_SIZE (e.g. 320) if it is too slow.
