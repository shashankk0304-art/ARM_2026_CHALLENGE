#!/usr/bin/env bash
# Install the bundled ARM64 Piper runtime and voice locally on the KV260.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
PY=.venv/bin/python
[[ -x "$PY" ]] || PY=python3
"$PY" -m pip install --no-index --find-links wheelhouse/speech --find-links wheelhouse -r requirements_speech.txt
for f in en_US-libritts_r-medium.onnx en_US-libritts_r-medium.onnx.json en_US-libritts_r-MODEL_CARD; do
  [[ -f "wheelhouse/speech/$f" ]] || { echo "Missing wheelhouse/speech/$f" >&2; exit 1; }
  cp "wheelhouse/speech/$f" models/
done
ASR_ZIP=wheelhouse/speech/vosk-model-small-en-us-0.15.zip
[[ -f "$ASR_ZIP" ]] || { echo "Missing $ASR_ZIP" >&2; exit 1; }
HI_ASR_ZIP=wheelhouse/speech/vosk-model-small-hi-0.22.zip
[[ -f "$HI_ASR_ZIP" ]] || { echo "Missing $HI_ASR_ZIP" >&2; exit 1; }
if [[ ! -d models/vosk-model-small-en-us-0.15 || ! -d models/vosk-model-small-hi-0.22 ]]; then
  "$PY" - "$ASR_ZIP" "$HI_ASR_ZIP" models <<'PY'
import sys, zipfile
from pathlib import Path
dest = Path(sys.argv[3]).resolve()
for archive in map(Path, sys.argv[1:3]):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = (dest / name).resolve()
            if target != dest and dest not in target.parents:
                raise SystemExit(f"Unsafe path in Vosk archive: {name}")
        z.extractall(dest)
PY
fi
echo 'Board-side English Piper speech output and English/Hindi Vosk speech recognition are installed.'
echo 'Retain the Piper voice card (CC BY 4.0) and Vosk model attribution (Apache 2.0).'
