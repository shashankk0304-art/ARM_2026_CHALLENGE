#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"
BOARD_HOME="${BOARD_HOME:-$HOME}"
YOLO_ONNX="${YOLO_ONNX:-models/ecosort_yolov8n_30class_v1.onnx}"
YOLO_LABELS="${YOLO_LABELS:-models/ecosort_yolov8n_30class_v1_labels.txt}"
LLAMA_RUNNER="${LLAMA_RUNNER:-$BOARD_HOME/executorch/cmake-out/examples/models/llama/llama_main}"
LLAMA_MODEL="${LLAMA_MODEL:-$BOARD_HOME/models/llama32_1b_xnnpack.pte}"
LLAMA_TOKENIZER="${LLAMA_TOKENIZER:-$BOARD_HOME/models/tokenizer.model}"

fail() { echo "EcoSort preflight failed: $*" >&2; exit 1; }
need_file() { [[ -s "$1" ]] || fail "missing file: $1"; }

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python3}"
[[ -x "$PYTHON" ]] || fail "missing .venv; run bash setup_kv260.sh"
need_file "$YOLO_ONNX"
need_file "$YOLO_LABELS"
need_file models/waste.onnx
need_file models/labels.txt
need_file i18n.json
need_file labels_map.json
need_file templates/index.html

[[ -x "$LLAMA_RUNNER" ]] || fail "ExecuTorch runner not found/executable: $LLAMA_RUNNER"
need_file "$LLAMA_MODEL"
need_file "$LLAMA_TOKENIZER"
command -v piper >/dev/null 2>&1 || fail "Piper missing; run bash setup_speech.sh"
need_file models/en_US-libritts_r-medium.onnx
[[ -d models/vosk-model-small-en-us-0.15 ]] || fail "English Vosk model missing; run bash setup_speech.sh"
[[ -d models/vosk-model-small-hi-0.22 ]] || fail "Hindi Vosk model missing; run bash setup_speech.sh"

"$PYTHON" - <<'PY'
import cv2, flask, numpy, vosk
print(f"Python deps OK | Flask {flask.__version__} | OpenCV {cv2.__version__} | NumPy {numpy.__version__} | Vosk OK")
PY

"$PYTHON" - "$YOLO_ONNX" <<'PY'
import cv2, sys
net = cv2.dnn.readNetFromONNX(sys.argv[1])
print("Main YOLO model loaded:", sys.argv[1])
PY

if command -v ss >/dev/null 2>&1 && ss -ltnH | awk '$4 ~ /:8000$/ { found=1 } END { exit !found }'; then
  fail "port 8000 is already listening. Check http://127.0.0.1:8000/api/health; do not start a duplicate server."
fi

echo "EcoSort preflight passed. Main detector: $YOLO_ONNX"
