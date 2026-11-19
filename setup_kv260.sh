#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt
.venv/bin/python - <<'PY'
import cv2, flask, numpy
print('Ready:', 'Flask', flask.__version__, '| OpenCV', cv2.__version__, '| NumPy', numpy.__version__)
PY
