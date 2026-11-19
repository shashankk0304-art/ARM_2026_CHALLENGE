#!/bin/bash
# Inside the Vitis-AI docker, after dpu_quantize.py test. Arch file path depends on your Vitis-AI version; use the KV260 arch.json.
vai_c_xir -x quantize_result/MobileNetV2_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o models -n waste
# -> models/waste.xmodel. Copy it with models/labels.txt to the board, boot a PYNQ-DPU image for KV260, and vision.py picks it up automatically.
