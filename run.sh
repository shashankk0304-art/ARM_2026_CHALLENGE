#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

# Defaults match the KV260 deployment. Override these when installing the
# ExecuTorch runner/PTE/tokenizer elsewhere on the board.
BOARD_HOME="${BOARD_HOME:-$HOME}"
export LLAMA_RUNNER="${LLAMA_RUNNER:-$BOARD_HOME/executorch/cmake-out/examples/models/llama/llama_main}"
export LLAMA_MODEL="${LLAMA_MODEL:-$BOARD_HOME/models/llama32_1b_xnnpack.pte}"
export LLAMA_TOKENIZER="${LLAMA_TOKENIZER:-$BOARD_HOME/models/tokenizer.model}"
export LLAMA_MAX_CTX="${LLAMA_MAX_CTX:-128}"
export LLAMA_MAX_NEW="${LLAMA_MAX_NEW:-40}"
export LLAMA_ECHO="${LLAMA_ECHO:-1}"

# Main detector: EcoSort's 30-class YOLOv8 ONNX model.
export YOLO_FORMAT="${YOLO_FORMAT:-ultralytics}"
export YOLO_ONNX="${YOLO_ONNX:-models/ecosort_yolov8n_30class_v1.onnx}"
export YOLO_LABELS="${YOLO_LABELS:-models/ecosort_yolov8n_30class_v1_labels.txt}"
export YOLO_SIZE="${YOLO_SIZE:-640}"
export YOLO_CONF="${YOLO_CONF:-0.35}"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project Python environment: $ROOT_DIR/.venv" >&2
  echo "Run: bash setup_kv260.sh" >&2
  exit 1
fi
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

bash "$ROOT_DIR/check_runtime.sh"
exec "$PYTHON" "$ROOT_DIR/app.py"
