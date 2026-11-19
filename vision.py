"""OpenCV vision: camera + classifier + natural-language description.
Runs on the KV260 Arm A53 CPU via cv2.dnn (swap in a Vitis-AI DPU model later
by replacing Classifier.predict - the rest of the app is unchanged)."""
import os, re, cv2, numpy as np
from waste_kb import map_label
MODEL  = os.environ.get("VISION_ONNX", "models/waste.onnx" if os.path.exists("models/waste.onnx") else "models/mobilenetv2-12.onnx")
LABELS = os.environ.get("VISION_LABELS", "models/labels.txt" if os.path.exists("models/labels.txt") else "models/imagenet_classes.txt")
CONF_MIN = float(os.environ.get("CONF_MIN", "0.30")); CUSTOM = os.path.basename(MODEL) == "waste.onnx" or os.path.exists("models/waste.xmodel")
COLORS = {"red":(0,0,200),"orange":(0,128,255),"yellow":(0,220,220),"green":(0,160,0),"blue":(200,80,0),
          "white":(235,235,235),"black":(25,25,25),"grey":(128,128,128),"brown":(40,75,120),"transparent/light":(210,220,200)}
def dominant_color(img):
    s = cv2.resize(img,(48,48)).reshape(-1,3).astype(np.float32)
    _,_,c = cv2.kmeans(s,1,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0),1,cv2.KMEANS_PP_CENTERS)
    return min(COLORS, key=lambda k: np.linalg.norm(c[0]-np.array(COLORS[k])))
def texture(img):
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); e = cv2.Canny(g,80,160).mean()/255
    return "smooth, shiny or plain" if e<0.04 else "textured or printed"
class Classifier:
    def __init__(self, model=MODEL, labels_path=LABELS):
        self.model_path = model
        self.net = cv2.dnn.readNetFromONNX(model) if os.path.exists(model) else None
        self.labels = [l.strip() for l in open(labels_path)] if os.path.exists(labels_path) else []
        self.material_model = set(self.labels) == {"can", "glass", "organic", "other", "paper", "plastic_bottle", "plastic_other"}
        self.wastewise_model = set(self.labels) == {"battery", "biological", "cardboard", "glass", "metal", "paper", "plastic", "trash"}
    def predict(self, img, k=5):
        if not self.net: return []
        if self.material_model:
            h, w = img.shape[:2]; side = min(h, w)
            y, x = (h-side)//2, (w-side)//2
            square = img[y:y+side, x:x+side]
            rgb = cv2.cvtColor(cv2.resize(square,(224,224)), cv2.COLOR_BGR2RGB).astype(np.float32)
            blob = np.ascontiguousarray((rgb / 127.5 - 1.0)[None], dtype=np.float32)  # published model expects NHWC [-1,1]
        elif self.wastewise_model:
            rgb = cv2.cvtColor(cv2.resize(img,(224,224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            blob = np.ascontiguousarray(rgb.transpose(2,0,1)[None], dtype=np.float32)
        else:
            rgb = cv2.cvtColor(cv2.resize(img,(224,224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            rgb = (rgb - np.array([0.485,0.456,0.406],np.float32)) / np.array([0.229,0.224,0.225],np.float32)
            blob = np.ascontiguousarray(rgb.transpose(2,0,1)[None], dtype=np.float32)
        self.net.setInput(blob); o = self.net.forward().flatten()
        if (self.material_model or self.wastewise_model) and np.all(o >= 0) and np.all(o <= 1) and .98 <= float(o.sum()) <= 1.02:
            p = o
        else:
            p = np.exp(o-o.max()); p/=p.sum()
        idx = p.argsort()[::-1][:k]
        return [(self.labels[i] if i<len(self.labels) else str(i), float(p[i])) for i in idx]
def describe_cls(img):
    """-> dict(description, category, conf, top) ; description is fed to Llama."""
    top = CLF.predict(img); cat, conf, obj = None, 0.0, "an unidentified object"
    for lab,p in top:
        if lab.strip().lower() == "other":
            continue  # open-set class is deliberately treated as unknown, not a material label
        c = map_label(lab)
        if c: cat, conf, obj = c, p, lab.split(",")[0]; break
    if not cat and top: obj = top[0][0].split(",")[0]; conf = top[0][1]
    material_min = float(os.environ.get("WASTE_CLASS_MIN", "0.55")) if (getattr(CLF, "material_model", False) or getattr(CLF, "wastewise_model", False)) else CONF_MIN
    if cat and CUSTOM and conf < material_min: cat = None
    if not cat:
        alt = _alt_classify(img)
        if alt: cat, conf, obj = alt[1], alt[2], alt[0]
    desc = f"The camera shows {obj} ({int(conf*100)}% confidence), mainly {dominant_color(img)} in colour, with a {texture(img)} surface."
    return dict(description=desc, category=cat, conf=conf, top=top)

# ---------- backend selection: DPU (KV260 FPGA) > CPU ----------
def _make_clf():
    if os.path.exists("models/waste.xmodel"):
        try:
            from dpu_backend import DpuClassifier
            c = DpuClassifier(); print("[vision] classifier running on KV260 DPU"); return c
        except Exception as e: print("[vision] DPU unavailable, falling back to CPU:", e)
    return Classifier()
CLF = _make_clf()
ALT_MODEL, ALT_LABELS = "models/wastewise_8class.onnx", "models/wastewise_8class_labels.txt"
ALT_CLF = Classifier(ALT_MODEL, ALT_LABELS) if os.path.exists(ALT_MODEL) and os.path.exists(ALT_LABELS) else None
def _alt_classify(img):
    if not ALT_CLF: return None
    pred = ALT_CLF.predict(img, k=1)
    if not pred or pred[0][0].strip().lower() in ("other", "trash"):
        return None
    label, conf = pred[0]
    if conf < float(os.environ.get("WASTE_ALT_MIN", "0.60")): return None
    cat = map_label(label)
    return (label, cat, conf) if cat else None

# ---------- YOLO multi-object detector (ONNX via cv2.dnn; Ultralytics YOLOv8/YOLO11 export) ----------
class Detector:
    def __init__(self):
        p = os.environ.get("YOLO_ONNX", "models/yolo.onnx"); lp = os.environ.get("YOLO_LABELS", "models/yolo_labels.txt")
        self.format = os.environ.get("YOLO_FORMAT", "ultralytics")
        self.conf = float(os.environ.get("YOLO_CONF", "0.35")); self.size = int(os.environ.get("YOLO_SIZE", "416"))
        self.model = None; self.net = None
        self.names = [l.strip() for l in open(lp)] if os.path.exists(lp) else []
        if os.path.exists(p):
            if self.format == "opencv_zoo":
                from yolox_backend import YoloX, COCO_CLASSES
                self.model = YoloX(p, self.conf, float(os.environ.get("YOLO_NMS", "0.5")))
                self.net = self.model.net; self.size = self.model.input_size
                if not self.names: self.names = list(COCO_CLASSES)
            else:
                self.net = cv2.dnn.readNetFromONNX(p)
    def detect(self, img):
        h, w = img.shape[:2]; s = min(self.size / w, self.size / h); nw, nh = int(w * s), int(h * s)
        if self.format == "opencv_zoo":
            canvas = np.full((self.size, self.size, 3), 114, np.uint8)
            canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
            dets = self.model.infer(rgb)
            result = []
            for x1, y1, x2, y2, conf, cid in dets:
                x1 = max(0, min(w, int(x1 / s))); y1 = max(0, min(h, int(y1 / s)))
                x2 = max(x1, min(w, int(x2 / s))); y2 = max(y1, min(h, int(y2 / s)))
                result.append(([x1, y1, x2-x1, y2-y1], self.names[int(cid)], float(conf)))
            return result
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8); canvas[dy:dy+nh, dx:dx+nw] = cv2.resize(img, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255., (self.size, self.size), swapRB=True))
        o = self.net.forward()[0].T                       # (N, 4+nc): cx, cy, w, h, class scores
        sc = o[:, 4:]; cid = sc.argmax(1); cf = sc.max(1); k = cf > self.conf; o, cid, cf = o[k], cid[k], cf[k]
        boxes = []
        for x, y, bw, bh in o[:, :4]:
            x1 = max(0, min(w, int((x - bw / 2 - dx) / s))); y1 = max(0, min(h, int((y - bh / 2 - dy) / s)))
            x2 = max(x1, min(w, int((x + bw / 2 - dx) / s))); y2 = max(y1, min(h, int((y + bh / 2 - dy) / s)))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        keep = np.array(cv2.dnn.NMSBoxes(boxes, cf.tolist(), self.conf, 0.45)).flatten() if boxes else []
        return [(boxes[i], self.names[cid[i]] if cid[i] < len(self.names) else str(cid[i]), float(cf[i])) for i in keep]
DET = Detector()
def describe_det(img):
    dets = sorted(DET.detect(img), key=lambda d: -d[2])[:6]; out = img.copy(); items = []
    for (x, y, bw, bh), lab, cf in dets:
        # Initialize the mapped category before testing whether this detector
        # label needs refinement. COCO e-waste labels should work directly.
        cat = map_label(lab)
        # COCO's generic "bottle" class cannot tell PET from glass. If a trained
        # waste classifier is installed, classify the crop to refine it. A bottle-only
        # classifier must not be applied to cans, food, phones, or other object types.
        has_custom_classifier = bool(getattr(CLF, "net", None)) or CLF.__class__.__name__ == "DpuClassifier"
        model_labels = [s.lower().replace("_", " ") for s in getattr(CLF, "labels", [])]
        bottle_only = bool(model_labels) and all("bottle" in s for s in model_labels)
        bottle_detection = bool(re.search(r"bottle|wine glass", lab, re.I))
        material_ambiguous = bool(getattr(CLF, "material_model", False) and re.search(r"bottle|wine glass|\bcup\b", lab, re.I))
        needs_refinement = cat is None or bottle_detection or material_ambiguous
        if CUSTOM and has_custom_classifier and needs_refinement and (not bottle_only or bottle_detection):
            crop = img[max(0,y):max(0,y)+bh, max(0,x):max(0,x)+bw]
            pred = CLF.predict(crop, k=1) if crop.size else []
            material_min = float(os.environ.get("WASTE_CLASS_MIN", "0.55")) if getattr(CLF, "material_model", False) else CONF_MIN
            if pred and pred[0][0].strip().lower() not in ("other", "trash") and pred[0][1] >= material_min and map_label(pred[0][0]):
                lab, cf = pred[0]
            elif cat is None:
                alt = _alt_classify(crop) if crop.size else None
                if alt: lab, cf = alt[0], alt[2]
        cat = map_label(lab); items.append(dict(label=lab, cat=cat, conf=round(cf, 2)))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (60, 180, 60) if cat else (0, 0, 220), 2)
        cv2.putText(out, f"{lab} {int(cf*100)}%", (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    # If COCO does not have this waste class (e.g., a drink can or a loose battery),
    # use the trained OpenCV waste classifier on the full image. This is an abstaining
    # fallback: it only reports labels with a mapped waste category and adequate confidence.
    if CUSTOM and not any(i["cat"] for i in items):
        try:
            pred = CLF.predict(img, k=1)
            if pred and pred[0][1] >= float(os.environ.get("VISION_FALLBACK_MIN", "0.55")):
                label, conf = pred[0]
                cat = map_label(label)
                if cat and label.strip().lower() not in ("other", "trash"):
                    items.append(dict(label=label, cat=cat, conf=round(conf, 2)))
            if not any(i["cat"] for i in items):
                alt = _alt_classify(img)
                if alt: items.append(dict(label=alt[0], cat=alt[1], conf=round(alt[2], 2)))
        except Exception as e:
            print("[vision] full-frame classifier fallback failed:", e)
    names = ", ".join(f"{i['label']} ({int(i['conf']*100)}%)" for i in items)
    recognized = next((i for i in items if i["cat"]), None)
    desc = f"The camera shows {len(items)} item(s): {names}." if items else "No waste item was confidently detected. Try a closer, well-lit photo with one item in frame."
    return dict(description=desc,
                category=recognized["cat"] if recognized else None, conf=recognized["conf"] if recognized else 0.0,
                top=[], items=items, annotated=out)
def describe(img):
    r = describe_det(img) if DET.net else describe_cls(img)
    if "items" not in r:
        r["items"] = [dict(label=r["description"].replace("The camera shows ", "").split(" (")[0], cat=r["category"], conf=round(r["conf"], 2))] if r["category"] else []
    r["backend"] = (("YOLO + waste classifier (CPU, OpenCV)" if CUSTOM else "YOLO (CPU, OpenCV)")
                     if DET.net else ("DPU (KV260 FPGA)" if CLF.__class__.__name__ == "DpuClassifier" else "MobileNetV2 (CPU, OpenCV)"))
    return r
