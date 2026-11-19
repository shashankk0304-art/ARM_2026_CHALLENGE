"""Run the Vitis-AI compiled classifier (models/waste.xmodel) on the KV260 DPU via PYNQ-DPU. Same predict() API as vision.Classifier.
Needs a PYNQ-DPU image for KV260 (dpu.bit next to this file, or set DPU_BIT). Not tested here: verify on your board."""
import os, numpy as np, cv2
class DpuClassifier:
    def __init__(self, xmodel="models/waste.xmodel", labels="models/labels.txt"):
        from pynq_dpu import DpuOverlay
        self.ov = DpuOverlay(os.environ.get("DPU_BIT", "dpu.bit")); self.ov.load_model(xmodel); self.r = self.ov.runner
        self.shin = tuple(self.r.get_input_tensors()[0].dims); self.shout = tuple(self.r.get_output_tensors()[0].dims)
        self.labels = [l.strip() for l in open(labels)]
    def predict(self, img, k=5):
        x = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        inp = [np.ascontiguousarray(x.reshape(self.shin), dtype=np.float32)]; out = [np.empty(self.shout, dtype=np.float32, order="C")]
        self.r.wait(self.r.execute_async(inp, out)); o = out[0].flatten(); p = np.exp(o - o.max()); p /= p.sum()
        return [(self.labels[i] if i < len(self.labels) else str(i), float(p[i])) for i in p.argsort()[::-1][:k]]
