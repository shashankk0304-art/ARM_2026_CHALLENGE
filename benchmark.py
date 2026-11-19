"""Repeatable CPU benchmark for the configured OpenCV object detector.

Run on KV260 from the project directory:
    .venv/bin/python benchmark.py /path/to/test.jpg --warmup 5 --repeat 30

This measures detector inference/postprocessing separately from the full vision pipeline. It does not
measure Llama or include image upload/network latency.
"""
import argparse
import os
import statistics
import time

# Match the current 30-class detector in run.sh when called directly.
os.environ.setdefault("YOLO_FORMAT", "ultralytics")
os.environ.setdefault("YOLO_ONNX", "models/ecosort_yolov8n_30class_v1.onnx")
os.environ.setdefault("YOLO_LABELS", "models/ecosort_yolov8n_30class_v1_labels.txt")
os.environ.setdefault("YOLO_SIZE", "640")
os.environ.setdefault("YOLO_CONF", "0.35")

import cv2
import vision


def summary(title, samples):
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, round(.95 * (len(ordered) - 1)))]
    print(f"{title}: n={len(samples)} min={min(samples):.2f} ms "
          f"median={statistics.median(samples):.2f} ms mean={statistics.mean(samples):.2f} ms "
          f"p95={p95:.2f} ms max={max(samples):.2f} ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Representative local image to benchmark")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=30)
    a = p.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        p.error(f"Could not read image: {a.image}")
    if not getattr(vision.DET, "net", None):
        p.error("YOLO detector is not loaded; check YOLO_ONNX and YOLO_FORMAT")
    print(f"model={os.environ['YOLO_ONNX']} format={vision.DET.format} labels={len(vision.DET.names)} backend=OpenCV-DNN CPU "
          f"input={img.shape[1]}x{img.shape[0]} detector_size={vision.DET.size} "
          f"opencv_threads={cv2.getNumThreads()} warmup={a.warmup} repeat={a.repeat}")
    for _ in range(max(0, a.warmup)):
        vision.DET.detect(img)
    samples = []
    for _ in range(max(1, a.repeat)):
        t0 = time.perf_counter_ns()
        vision.DET.detect(img)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    summary("detector inference+postprocess", samples)
    print(f"throughput={1000.0 / statistics.mean(samples):.2f} frames/s (single-thread serial benchmark)")

    total = []
    for _ in range(min(10, max(1, a.repeat))):
        t0 = time.perf_counter_ns()
        result = vision.describe(img)
        total.append((time.perf_counter_ns() - t0) / 1e6)
    summary("full vision.describe (detector + classifier + annotation)", total)
    print(f"last_category={result['category']} last_description={result['description']}")


if __name__ == "__main__":
    main()
