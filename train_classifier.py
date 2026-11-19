"""Train on ANY image-folder dataset (data/<class_name>/*.jpg) -> models/waste.onnx + models/labels.txt.
Run on a PC/laptop (GPU optional), then copy models/ to the KV260.   python3 train_classifier.py data 10
Works with TrashNet, Kaggle Garbage-12, TACO crops, or your own photos - class folder names are mapped by labels_map.json."""
import sys, os, random, copy, torch, torch.nn as nn
from torchvision import datasets, models, transforms as T
from torch.utils.data import DataLoader
from waste_kb import map_label
root = sys.argv[1]; epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
tr_t = T.Compose([T.RandomResizedCrop(224,(0.6,1)),T.RandomHorizontalFlip(),T.ColorJitter(.3,.3,.3),T.ToTensor(),N])
ev_t = T.Compose([T.Resize((224,224)),T.ToTensor(),N])
train_data = datasets.ImageFolder(root, tr_t)
eval_data = datasets.ImageFolder(root, ev_t); classes = train_data.classes
if eval_data.classes != classes: raise RuntimeError("Training and validation class folders differ")
for c in classes: print(f"class '{c}' -> {map_label(c) or 'UNMAPPED (add it to labels_map.json)'}")
torch.manual_seed(0); random.seed(0)
# Keep all crops from the same source photograph together to prevent train/val/test leakage.
groups = {}
for i, (path, class_id) in enumerate(train_data.samples):
    stem = os.path.basename(path).split("__ann", 1)[0]
    groups.setdefault(stem, []).append((i, class_id))
group_names = sorted(groups); random.shuffle(group_names)
if len(group_names) < 3: raise RuntimeError("Need at least three independent source images for train/validation/test")
class_groups = {class_id: [g for g in group_names if any(cid == class_id for _, cid in groups[g])]
                for class_id in range(len(classes))}
val_groups = set()
test_groups = set()
for class_id, name in enumerate(classes):
    candidates = class_groups[class_id][:]
    random.shuffle(candidates)
    if len(candidates) < 3:
        raise RuntimeError(f"Class '{name}' needs at least three independent source images for train/validation/test")
    target = max(1, round(.15 * len(candidates)))
    for g in candidates:
        if sum(any(cid == class_id for _, cid in groups[x]) for x in val_groups) >= target: break
        val_groups.add(g)
for class_id, name in enumerate(classes):
    candidates = class_groups[class_id][:]
    random.shuffle(candidates)
    target = max(1, round(.15 * len(candidates)))
    for g in candidates:
        if g in val_groups: continue
        if sum(any(cid == class_id for _, cid in groups[x]) for x in test_groups) >= target: break
        test_groups.add(g)
train_groups = set(group_names) - val_groups - test_groups
for class_id, name in enumerate(classes):
    if not any(cid == class_id for g in train_groups for _, cid in groups[g]) or \
       not any(cid == class_id for g in val_groups for _, cid in groups[g]) or \
       not any(cid == class_id for g in test_groups for _, cid in groups[g]):
        raise RuntimeError(f"Not enough independent source photos for train/validation/test in class '{name}'")
tr_idx = [i for g in train_groups for i, _ in groups[g]]
va_idx = [i for g in val_groups for i, _ in groups[g]]
te_idx = [i for g in test_groups for i, _ in groups[g]]
tr_set = torch.utils.data.Subset(train_data, tr_idx)
va_set = torch.utils.data.Subset(eval_data, va_idx)
te_set = torch.utils.data.Subset(eval_data, te_idx)
nv = len(va_idx)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = models.mobilenet_v2(weights="DEFAULT"); m.classifier[1] = nn.Linear(1280, len(classes)); m.to(dev)
for p in m.features.parameters(): p.requires_grad = False
counts = [sum(train_data.samples[i][1] == c for i in tr_idx) for c in range(len(classes))]
if not all(counts): raise RuntimeError("Every class needs training images")
weights = torch.tensor([len(tr_idx)/(len(classes)*n) for n in counts], dtype=torch.float32, device=dev)
opt = torch.optim.Adam(m.classifier.parameters(), 1e-3); lossf = nn.CrossEntropyLoss(weight=weights)
workers = 0 if os.name == "nt" else 2
tl = DataLoader(tr_set, 32, shuffle=True, num_workers=workers, generator=torch.Generator().manual_seed(0)); vl = DataLoader(va_set, 64, num_workers=workers); test = DataLoader(te_set, 64, num_workers=workers)
best_acc = -1; best_state = None; unfreeze_at = max(1, epochs // 2)
for ep in range(epochs):
    if ep == unfreeze_at:
        for block in list(m.features.children())[-3:]:
            for p in block.parameters(): p.requires_grad = True
        opt = torch.optim.Adam([{"params": [p for p in m.features.parameters() if p.requires_grad], "lr": 1e-5},
                                {"params": m.classifier.parameters(), "lr": 1e-4}])
    m.train()
    for x, y in tl: opt.zero_grad(); lossf(m(x.to(dev)), y.to(dev)).backward(); opt.step()
    m.eval(); ok = 0
    with torch.no_grad():
        for x, y in vl: ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item()
    val_acc = ok/nv
    print(f"epoch {ep+1}/{epochs} val_acc {val_acc:.3f}")
    if val_acc > best_acc: best_acc, best_state = val_acc, copy.deepcopy(m.state_dict())
m.load_state_dict(best_state)
m.eval(); ok = 0; nt = 0
confusion = [[0 for _ in classes] for _ in classes]
with torch.no_grad():
    for x, y in test:
        pred = m(x.to(dev)).argmax(1).cpu()
        ok += (pred == y).sum().item(); nt += y.numel()
        for actual, guessed in zip(y.tolist(), pred.tolist()): confusion[actual][guessed] += 1
print(f"held_out_test_acc {ok/nt:.3f} ({ok}/{nt})")
for c, name in enumerate(classes):
    denom = sum(confusion[c])
    print(f"held_out_test_recall {name}: {confusion[c][c]/denom:.3f} ({confusion[c][c]}/{denom})")
os.makedirs("models", exist_ok=True)
torch.onnx.export(m.eval().cpu(), torch.randn(1,3,224,224), "models/waste.onnx", input_names=["input"], output_names=["logits"], opset_version=12)
open("models/labels.txt","w").write("\n".join(classes)); print("saved models/waste.onnx and models/labels.txt")
torch.save(m.state_dict(), "models/waste.pth")   # used by dpu_quantize.py
