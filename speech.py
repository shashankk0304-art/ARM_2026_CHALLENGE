"""Optional fully local speech I/O for KV260 (Vosk ASR + Piper TTS)."""
import io
import json
import os
import subprocess
import tempfile
import threading
import wave

ROOT = os.path.dirname(os.path.abspath(__file__))
VOSK_MODELS = {
    "en": os.environ.get("VOSK_MODEL", os.path.join(ROOT, "models", "vosk-model-small-en-us-0.15")),
    "hi": os.environ.get("VOSK_MODEL_HI", os.path.join(ROOT, "models", "vosk-model-small-hi-0.22")),
}
PIPER = os.environ.get("PIPER", "piper")
PIPER_MODEL = os.environ.get("PIPER_MODEL", os.path.join(ROOT, "models", "en_US-libritts_r-medium.onnx"))
ASR_LOCK = threading.Lock()
_models = {}
_lock = threading.Lock()


def transcribe(wav_bytes, lang="en"):
    """Transcribe mono, 16-bit PCM WAV offline on the board for supported languages."""
    lang = str(lang or "en").lower().split("-")[0].split("_")[0]
    model_path = VOSK_MODELS.get(lang)
    if model_path is None:
        raise RuntimeError(f"Offline speech recognition is not configured for language '{lang}'.")
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError as e:
        raise RuntimeError("Offline speech recognition is not installed (Vosk).") from e
    if not os.path.isdir(model_path):
        raise RuntimeError(f"Vosk {lang} model folder not found: {model_path}")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                raise ValueError("Audio must be mono 16 kHz PCM. Use the microphone button in the website.")
            pcm = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError("Audio upload is not a valid WAV file.") from e
    with ASR_LOCK:
        if lang not in _models: _models[lang] = Model(model_path)
        recognizer = KaldiRecognizer(_models[lang], 16000)
        recognizer.AcceptWaveform(pcm)
        result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()


def synthesize(text):
    """Create a WAV using a local Piper binary and voice; no browser/cloud TTS."""
    if not os.path.isfile(PIPER_MODEL):
        raise RuntimeError(f"Piper voice model not found: {PIPER_MODEL}")
    with tempfile.NamedTemporaryFile(prefix="ecosort-", suffix=".wav", delete=False) as f:
        out_path = f.name
    try:
        try:
            proc = subprocess.run([PIPER, "--model", PIPER_MODEL, "--output_file", out_path],
                                  input=text, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=90, check=False)
        except FileNotFoundError as e:
            raise RuntimeError(f"Piper executable not found: {PIPER}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Local speech synthesis timed out.") from e
        if proc.returncode != 0:
            raise RuntimeError("Piper failed: " + proc.stderr[-600:])
        with open(out_path, "rb") as f: return io.BytesIO(f.read())
    finally:
        try: os.unlink(out_path)
        except OSError: pass
