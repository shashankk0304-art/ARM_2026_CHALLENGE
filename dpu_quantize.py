"""Quantize the trained classifier for the DPU. Run INSIDE the Vitis-AI PyTorch docker on a PC (replaces the old dummy quantize.py).
Needs models/waste.pth + models/labels.txt (from train_classifier.py) and ~100+ calibration images in calib/.
Step 1: python3 dpu_quantize.py calib      Step 2: python3 dpu_quantize.py test  -> quantize_result/*.xmodel (then dpu_compile.sh)"""
import sys, glob, torch, torch.nn as nn, cv2, numpy as np
from torchvision import models
from pytorch_nndct.apis import torch_quantizer
mode = sys.argv[1]; n = len(open("models/labels.txt").read().split("\n"))
m = models.mobilenet_v2(); m.classifier[1] = nn.Linear(1280, n); m.load_state_dict(torch.load("models/waste.pth", map_location="cpu")); m.eval()
def prep(f):
    x = cv2.cvtColor(cv2.resize(cv2.imread(f), (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return torch.from_numpy(((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32)).permute(2, 0, 1)[None]
dummy = torch.randn(1, 3, 224, 224); q = torch_quantizer(mode, m, (dummy,), device=torch.device("cpu")); qm = q.quant_model
if mode == "calib":
    with torch.no_grad():
        for f in sorted(glob.glob("calib/*.jpg"))[:200]: qm(prep(f))
    q.export_quant_config()
else:
    with torch.no_grad(): qm(dummy)
    q.export_xmodel(output_dir="quantize_result", deploy_check=False)
