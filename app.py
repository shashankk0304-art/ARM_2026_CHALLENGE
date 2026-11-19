import os, cv2, json, time, threading, base64, glob, re, shutil, importlib.util
import numpy as np
from flask import Flask, Response, jsonify, request, render_template, send_file
import vision, llm_executorch as llm
import speech
from waste_kb import KB, map_label
app = Flask(__name__, template_folder="templates")
TR = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n.json"), encoding="utf-8"))
NATIVE_TERMS = {
 "battery": ["बैटरी", "बैटरियां", "बैटरियाँ", "బ్యాటరీ", "బ్యాటరీలు", "ಬ್ಯಾಟರಿ", "ಬ್ಯಾಟರಿಗಳು", "बॅटरी", "बॅटऱ्या"],
 "ewaste": ["ई-कचरा", "ई कचरा", "इलेक्ट्रॉनिक कचरा", "ఈ-వ్యర్థ", "ఈ వ్యర్థ", "ఎలక్ట్రానిక్", "ಇ-ತ್ಯಾಜ್ಯ", "ಎಲೆಕ್ಟ್ರಾನಿಕ್", "मोबाइल", "मोबाईल", "फोन", "चार्जर", "लैपटॉप", "ల్యాప్‌టాప్", "మొబైల్", "ఫోన్", "ఛార్జర్", "ಲ್ಯಾಪ್‌ಟಾಪ್", "ಮೊಬೈಲ್", "ಫೋನ್", "ಚಾರ್ಜರ್"],
 "hazardous": ["खतरनाक", "रसायन", "दवाई", "రసాయన", "ప్రమాదకర", "ಅಪಾಯಕಾರಿ", "ರಾಸಾಯನಿಕ", "धोकादायक", "औषध", "रासायनिक"],
 "plastic": ["प्लास्टिक", "ప్లాస్టిక్", "ಪ್ಲಾಸ್ಟಿಕ್"],
 "paper": ["कागज", "कागज़", "पेपर", "కాగితం", "ಕಾಗದ", "कार्डबोर्ड", "కార్డ్‌బోర్డ్", "ಕಾರ್ಡ್‌ಬೋರ್ಡ್", "कागद", "पुठ्ठा"],
 "glass": ["कांच", "काँच", "शीशा", "గాజు", "ಗಾಜು", "काच", "काचे"],
 "metal": ["धातु", "एल्युमिनियम", "एल्यूमिनियम", "లోహం", "అల్యూమినియం", "ಲೋಹ", "ಅಲ್ಯೂಮಿನಿಯಂ", "धातू", "अॅल्युमिनियम"],
 "organic": ["जैविक", "खाद्य", "खाना", "సేంద్రీయ", "ఆహార", "ಸಾವಯವ", "ಆಹಾರ", "सेंद्रिय", "अन्नकचरा"],
 "textile": ["कपड़े", "वस्त्र", "బట్టలు", "ಬಟ್ಟೆ", "ವಸ್ತ್ರ", "कपडे", "कापड"],
 "styrofoam": ["थर्माकोल", "थर्मोकोल", "फोम", "థర్మోకోల్", "ఫోమ్", "ಥರ್ಮೋಕೋಲ್", "ಫೋಮ್"],
}
def query_category(q):
    cat = map_label(q)
    if cat: return cat
    low = q.casefold()
    for cat in ("battery", "ewaste", "hazardous", "plastic", "paper", "glass", "metal", "organic", "textile", "styrofoam"):
        if any(term.casefold() in low for term in NATIVE_TERMS[cat]): return cat
    return None
CAM_SETTING = os.environ.get("CAM_INDEX", "auto").strip().lower()
CAM = int(CAM_SETTING) if CAM_SETTING.lstrip("-").isdigit() else None
lock = threading.Lock(); frame = None; active_cam = None
def grab():
    global frame, active_cam
    while True:
        if CAM is not None and CAM >= 0:
            indices = [CAM]
        else:
            devs = sorted(glob.glob("/dev/video[0-9]*"), key=lambda p: int(re.search(r"(\d+)$", p).group(1)))
            indices = [int(re.search(r"(\d+)$", p).group(1)) for p in devs]
            if not indices:
                active_cam = None
                with lock: frame = None
                time.sleep(2)
                continue
        connected = False
        for index in indices:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            ok, f = cap.read()
            if not ok:
                cap.release(); continue
            active_cam = index; connected = True
            with lock: frame = f
            while True:
                ok, f = cap.read()
                if not ok: break
                with lock: frame = f
            cap.release(); active_cam = None
            break
        if not connected:
            with lock: frame = None
        time.sleep(2)
threading.Thread(target=grab,daemon=True).start()
@app.route("/")
def index(): return render_template("index.html", cats={k:v["name"] for k,v in KB.items()}, tr=TR)
@app.route("/video")
def video():
    def gen():
        while True:
            with lock: f = None if frame is None else frame.copy()
            if f is not None:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n"+cv2.imencode(".jpg",f,[1,70])[1].tobytes()+b"\r\n"
            time.sleep(0.07)
    return Response(gen(),mimetype="multipart/x-mixed-replace; boundary=f")
@app.route("/api/capture",methods=["POST"])
def capture():
    upload = request.files.get("image")
    if upload:
        raw = upload.read()
        f = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR) if raw else None
    else:
        with lock: f = None if frame is None else frame.copy()
    if f is None:
        msg = ("Uploaded photo could not be decoded; try JPEG or PNG." if upload else
               "No KV260 camera frame is available. Use the phone photo button or connect a USB camera to the board.")
        return jsonify(error=msg),400
    try:
        r = vision.describe(f)
        a = r.pop("annotated", f)
        r["thumb"] = base64.b64encode(cv2.imencode(".jpg",cv2.resize(a,(480,360)))[1]).decode()
    except Exception:
        app.logger.exception("OpenCV image analysis failed")
        return jsonify(error="OpenCV could not analyse this image. Check /home/ubuntu/ecosort/app.log for the board error."), 500
    return jsonify(r)
@app.route("/api/advise")
def advise():
    cat = request.args.get("cat"); desc = request.args.get("desc","")
    if cat not in KB: return jsonify(error="unknown category"),400
    kb = KB[cat]; COUNTS[cat] = COUNTS.get(cat, 0) + 1; lang = request.args.get("lang", "en")
    if lang not in ("en", "hi", "te", "kn", "mr"): lang = "en"
    tr = TR.get(cat, {}).get(lang)
    def ev(d): return "data: "+json.dumps(d)+"\n\n"
    def gen():
        yield ev(dict(sec="kb",tr=TR.get(cat,{}).get(lang),**kb))
        try:
            for item in llm.stream(desc,cat,kb,lang):
                if item[0]=="delta": yield ev(dict(sec="delta",text=item[1]))
                else:
                    fallback = f"{tr['name']} — {tr['bin']}." if tr else f"{kb['impact']} Tip: {kb['tip']}"
                    yield ev(dict(sec="final",text=item[1] or fallback,src=item[2] if item[1] else "kb"))
        except Exception as e:
            app.logger.exception("Llama advice failed")
            fallback = f"{tr['name']} — {tr['bin']}." if tr else f"{kb['impact']} Tip: {kb['tip']}"
            yield ev(dict(sec="final",text=fallback,src="kb"))
        yield ev(dict(sec="done"))
    return Response(gen(),mimetype="text/event-stream")
COUNTS = {}
def ev(d): return "data: " + json.dumps(d) + "\n\n"
def chat_answer(model_advice, cat, kb, lang):
    model_advice = re.sub(r"^\s*(?:\*\*)?Advice(?:\*\*)?:\s*", "", model_advice or "", flags=re.I).strip()
    if kb:
        localized = TR.get(cat, {}).get(lang) or {"name": kb["name"], "bin": kb["bin"]}
        advice = model_advice or f"Put {localized['name']} in {localized['bin']}."
        reason = kb["impact"]
        learn = f"{kb['recycle']} Tip: {kb['tip']}"
    else:
        advice = model_advice or "Keep batteries and electronics out of mixed household recycling; use a dedicated collection route."
        reason = "Incorrect disposal can release harmful materials or cause fires in waste collection."
        learn = "Separate clean recyclables, and check your municipality's collection guidance."
    return f"Advice: {advice}\nReasoning: {reason}\nLearn: {learn}"
def chat_fallback(cat, kb, lang): return chat_answer("", cat, kb, lang)
@app.route("/api/stats")
def stats(): return jsonify(total=sum(COUNTS.values()), by={KB[k]["name"].split(" ")[0]: v for k, v in COUNTS.items()})
@app.route("/api/health")
def health():
    with lock: camera_ready = frame is not None
    classifier_ready = bool(getattr(getattr(vision, "CLF", None), "net", None)) or vision.CLF.__class__.__name__ == "DpuClassifier"
    speech_output_ready = bool(shutil.which(speech.PIPER) and os.path.isfile(speech.PIPER_MODEL))
    speech_languages = [lang for lang, path in speech.VOSK_MODELS.items() if os.path.isdir(path)] if importlib.util.find_spec("vosk") else []
    speech_input_ready = bool(speech_languages)
    return jsonify(camera=camera_ready, camera_index=active_cam,
                   camera_hint=None if camera_ready else "Use the phone photo button or attach a USB camera to the KV260",
                   detector=bool(getattr(vision.DET, "net", None)),
                   classifier=classifier_ready, llama_runner=os.path.isfile(llm.RUNNER),
                   llama_model=os.path.isfile(llm.MODEL), tokenizer=os.path.isfile(llm.TOK),
                   speech_output=speech_output_ready, speech_input=speech_input_ready,
                   speech_languages=speech_languages)
@app.route("/api/ask")
def ask():
    q = request.args.get("q", "").strip()[:300]
    if not q: return jsonify(error="empty"), 400
    cat = query_category(q); kb = KB.get(cat); lang = request.args.get("lang", "en")
    if lang not in ("en", "hi", "te", "kn", "mr"): lang = "en"
    def gen():
        try:
            for it in llm.ask(q, cat, kb, lang):
                if it[0] == "delta": yield ev(dict(sec="delta", text=it[1]))
                else:
                    yield ev(dict(sec="final", text=chat_answer(it[1], cat, kb, lang), src="llama-grounded" if it[1] else "kb"))
        except Exception:
            app.logger.exception("Llama chat failed")
            yield ev(dict(sec="final", text=chat_fallback(cat, kb, lang), src="kb"))
    return Response(gen(), mimetype="text/event-stream")
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if not request.data: return jsonify(error="No audio received"),400
    lang = request.args.get("lang", "en")
    try: return jsonify(text=speech.transcribe(request.data, lang))
    except RuntimeError as e: return jsonify(error=str(e)),503
    except ValueError as e: return jsonify(error=str(e)),400

@app.route("/api/speak", methods=["POST"])
def speak():
    text = str((request.get_json(silent=True) or {}).get("text", "")).strip()[:1200]
    if not text: return jsonify(error="No text received"),400
    try: audio = speech.synthesize(text)
    except RuntimeError as e: return jsonify(error=str(e)),503
    return send_file(audio, mimetype="audio/wav", download_name="ecosort.wav", as_attachment=False)
import random
@app.route("/api/quiz")
def quiz():
    """5 questions auto-generated from the knowledge base (no LLM needed, instant, always correct)."""
    ks = list(KB); qs = []
    for k in random.sample(ks, 5):
        if random.random() < .5:
            field, q = "bin", f"Which bin/route is correct for: {KB[k]['name']}?"
        else:
            field, q = "decomp", f"Approximately how long does it take to decompose: {KB[k]['name']}?"
        wrong = list({KB[x][field] for x in ks if KB[x][field] != KB[k][field]}); random.shuffle(wrong)
        opts = wrong[:3] + [KB[k][field]]; random.shuffle(opts)
        qs.append(dict(q=q, options=opts, answer=opts.index(KB[k][field]), why=f"{KB[k]['impact']} Tip: {KB[k]['tip']}"))
    return jsonify(qs)
if __name__=="__main__":
    cert, key = os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_KEY_FILE")
    ssl = (cert, key) if cert and key and os.path.isfile(cert) and os.path.isfile(key) else None
    app.run(host="0.0.0.0",port=8000,threaded=True,ssl_context=ssl)
