"""Safely merge public, labeled image-folder archives for EcoSort training.

Inputs: ETID-v2 (e-waste, CC BY 4.0) and UCI RealWaste (household waste, CC BY 4.0).
Download the archives from the links in data/README.md, then run this script.
Only image files in labeled directories are copied; ZIP paths are never extracted.
"""
import argparse
import hashlib
import re
import shutil
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALIASES = {
    "battery": "battery", "battery waste": "battery", "batteries": "battery",
    "cable": "cable", "cables": "cable", "wire": "cable", "wires": "cable",
    "keyboard": "keyboard", "laptop": "laptop", "mobile": "mobile_phone",
    "mobile phone": "mobile_phone", "smartphone": "mobile_phone", "phone": "mobile_phone",
    "mouse": "mouse", "pcb": "pcb", "printed circuit board": "pcb",
    "cardboard": "cardboard", "paper": "paper", "glass": "glass", "metal": "metal",
    "plastic": "plastic", "food organics": "organic", "food organic": "organic",
    "organic": "organic", "vegetation": "organic", "textile trash": "textile",
    "textile": "textile", "miscellaneous trash": "miscellaneous_trash",
}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", s.strip().lower()))


def class_for(member):
    parts = Path(member).parts[:-1]
    for part in reversed(parts):
        label = norm(part)
        value = ALIASES.get(label)
        if value:
            return value
        # Be tolerant of archive folder names such as "Mobile-Phone-Waste",
        # "food_organics", or previously unseen ETID subtypes.
        if "battery" in label:
            return "battery"
        if any(term in label for term in ("paper", "cardboard")):
            return "paper" if "cardboard" not in label else "cardboard"
        if "glass" in label:
            return "glass"
        if any(term in label for term in ("plastic", "polymer")):
            return "plastic"
        if any(term in label for term in ("metal", "aluminium", "aluminum", "steel", "tin")):
            return "metal"
        if any(term in label for term in ("organic", "vegetation", "food", "biological")):
            return "organic"
        if any(term in label for term in ("textile", "fabric", "clothing")):
            return "textile"
        if any(term in label for term in ("ewaste", "e waste", "electronic", "laptop", "keyboard", "mouse",
                                          "mobile", "smartphone", "phone", "circuit", "pcb", "cable", "charger",
                                          "powerbank", "hard disk", "monitor", "printer", "router", "tablet",
                                          "earphone", "headphone", "appliance")):
            return "ewaste"
        if any(term in label for term in ("miscellaneous", "residual", "trash")):
            return "miscellaneous_trash"
    return None


def ingest(zip_path, dataset_name, output):
    count = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            if info.is_dir() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = class_for(info.filename)
            if not label:
                continue
            # Include source path in the filename so identical basenames from different
            # directories do not overwrite each other and source-image splits stay traceable.
            digest = hashlib.sha1(info.filename.encode("utf-8")).hexdigest()[:10]
            target_dir = output / label
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{dataset_name}__{digest}__{path.name}"
            if target.exists():
                continue
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count[label] = count.get(label, 0) + 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etid-zip", type=Path, required=True, help="ETID-v2 ZIP archive")
    p.add_argument("--realwaste-zip", type=Path, required=True, help="UCI RealWaste ZIP archive")
    p.add_argument("--output", type=Path, default=Path("data/public_waste_training"))
    a = p.parse_args()
    if not a.etid_zip.is_file() or not a.realwaste_zip.is_file():
        p.error("Both dataset archives must exist before preparation.")
    a.output.mkdir(parents=True, exist_ok=True)
    all_counts = {}
    for archive, name in ((a.etid_zip, "ETID"), (a.realwaste_zip, "RealWaste")):
        counts = ingest(archive, name, a.output)
        print(f"{name}: {sum(counts.values())} images -> {counts}")
        for label, n in counts.items():
            all_counts[label] = all_counts.get(label, 0) + n
    (a.output / "DATA_SOURCES.txt").write_text(
        "ETID-v2: https://data.mendeley.com/datasets/bndsgcspbx/2 (CC BY 4.0; cite Sayed et al., DOI 10.17632/bndsgcspbx.2)\n"
        "RealWaste: https://archive.ics.uci.edu/dataset/908/realwaste (CC BY 4.0; cite the dataset paper)\n"
        "Image filenames preserve source dataset and archive member hash for grouped train/validation/test splits.\n",
        encoding="utf-8")
    print(f"Combined: {sum(all_counts.values())} images in {len(all_counts)} classes at {a.output}")
    print("Next: python train_classifier.py data/public_waste_training 10")


if __name__ == "__main__":
    main()
