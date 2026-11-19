"""Extract licensed PET/plastic-bottle and glass-bottle crops from official TACO COCO annotations.

Usage: python prepare_taco_crops.py --root path/to/TACO --output data/taco_crops
The TACO repository's download.py should have populated root/data and root/data/annotations.json.
Only explicitly CC0, CC BY, or CC BY-SA images are included; per-image attribution is written to CSV.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2


def reusable_license(license_row):
    if not license_row:
        return False
    text = (str(license_row.get("name", "")) + " " + str(license_row.get("url", ""))).lower()
    # Do not use NC/ND assets for a challenge entry; require explicit reusable license.
    if "nc" in text or "nd" in text or "no known copyright" in text:
        return False
    return "creativecommons.org/publicdomain/zero" in text or bool(
        re.search(r"creativecommons\.org/licenses/by(?:-sa)?(?:/|\b)", text)
    )


def target_for(name):
    n = name.lower().replace("_", " ")
    if "plastic" in n and "bottle" in n:
        return "plastic_bottle"
    if "glass" in n and "bottle" in n:
        return "glass_bottle"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="TACO checkout/dataset root")
    ap.add_argument("--output", type=Path, default=Path("data/taco_crops"))
    args = ap.parse_args()
    ann_path = args.root / "data" / "annotations.json"
    if not ann_path.is_file():
        raise SystemExit(f"Missing {ann_path}; download the dataset using the official TACO instructions first.")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    licenses = {int(x["id"]): x for x in coco.get("licenses", [])}
    images = {int(x["id"]): x for x in coco.get("images", [])}
    cats = {int(x["id"]): x["name"] for x in coco.get("categories", [])}
    args.output.mkdir(parents=True, exist_ok=True)
    kept, rejected, missing = 0, 0, 0
    with (args.output / "ATTRIBUTION.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["crop", "source_image", "license", "license_url", "taco_category"])
        for obj in coco.get("annotations", []):
            cat = cats.get(int(obj.get("category_id", -1)), "")
            target = target_for(cat)
            if not target or obj.get("iscrowd", 0):
                continue
            image = images.get(int(obj.get("image_id", -1)))
            if not image:
                missing += 1; continue
            lic = licenses.get(int(image.get("license", -1)))
            if not reusable_license(lic):
                rejected += 1; continue
            image_path = args.root / "data" / image["file_name"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                missing += 1; continue
            x, y, w, h = [float(v) for v in obj.get("bbox", [0, 0, 0, 0])]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            crop = frame[y1:y2, x1:x2]
            out_dir = args.output / target
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"src{int(image['id']):06d}__ann{int(obj['id']):06d}.jpg"
            if cv2.imwrite(str(out_dir / out_name), crop):
                writer.writerow([out_name, image["file_name"], lic.get("name", ""), lic.get("url", ""), cat])
                kept += 1
    print(f"Saved {kept} licensed bottle crops to {args.output}; excluded {rejected} unapproved-license annotations; missing/unreadable {missing}.")
    print("Review class counts and keep the source-image groups separate between train and validation.")


if __name__ == "__main__":
    main()
