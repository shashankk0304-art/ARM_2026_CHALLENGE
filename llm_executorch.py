"""ExecuTorch Llama wrapper. New-token requests are clamped by the exported PTE context.
The current board PTE was exported at 128 tokens, so prompts are kept short and the UI
shows the complete curated facts independently. Re-exporting to 2048 is needed for longer answers.
"""
import os, re, subprocess, threading, queue
RUNNER = os.environ.get("LLAMA_RUNNER", "./cmake-out/examples/models/llama/llama_main")
MODEL  = os.environ.get("LLAMA_MODEL", "llama3_2_1b.pte")
TOK    = os.environ.get("LLAMA_TOKENIZER", "tokenizer.model")
MAXCTX = int(os.environ.get("LLAMA_MAX_CTX", "128"))   # must be <= exported max_context_len
TEMP   = os.environ.get("LLAMA_TEMP", "0.3")
TIMEOUT= int(os.environ.get("LLAMA_TIMEOUT", "240"))
_lock = threading.Lock()   # one inference at a time on the board
SYS = "EcoSort educator. Use given facts; never invent local rules."
def _tmpl(user):  # runner adds BOS itself, so no <|begin_of_text|> here
    return ("<|start_header_id|>system<|end_header_id|>\n\n"+SYS+"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            +user+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
def _clean(raw, prompt):
    if prompt in raw: raw = raw.split(prompt,1)[1]
    raw = re.sub(r"PyTorchObserver.*","",raw); raw = re.sub(r"<\|[^>]*\|>","",raw)
    raw = "\n".join(l for l in raw.splitlines() if not re.match(r"^(I \d+|E \d+|Prefill|Generated|\[)",l.strip()))
    return raw.strip()
_SENTENCE_END = r"[.!?।॥！？。]"
def _complete(t): return bool(re.search(_SENTENCE_END + r"[\"')\]]?$", t))
def _trim(t):
    if _complete(t): return t
    m = list(re.finditer(_SENTENCE_END, t)); return t[:m[-1].end()] if m else t
def _is_irrelevant_refusal(t):
    """Reject generic safety refusals for ordinary recycling questions; caller uses grounded KB facts."""
    s = re.sub(r"^[\s\W]+", "", t).lower()
    return (s.startswith(("i can't assist", "i cannot assist", "i can't help", "i cannot help",
                          "i am unable to help", "i'm unable to help", "i am unable to assist"))
            or "promotes recycling harmful practices" in s
            or "responsible recycling practices instead" in s)
def run(user_prompt, max_new=110):
    out = ""
    for attempt in range(2):
        prompt = _tmpl(user_prompt)
        cmd = [RUNNER,"--model_path",MODEL,"--tokenizer_path",TOK,"--prompt",prompt,"--max_new_tokens",str(max_new),"--temperature",TEMP]
        with _lock:
            p = subprocess.run(cmd,capture_output=True,text=True,timeout=TIMEOUT)
        out = _clean(p.stdout+"\n"+p.stderr if not p.stdout else p.stdout, prompt)
        if out and _complete(out): return out
        max_new += 80
    return _trim(out)
def advise(desc, cat, kb):
    """Yield (section, text, source) one section at a time."""
    ctx = f"Observation: {desc} Category: {kb['name']}. Bin: {kb['bin']}."
    jobs = [
     ("identify", f"{ctx} In 2 sentences, tell the user what this waste is and which bin it goes in."),
     ("dispose",  f"{ctx} Steps: {'; '.join(kb['steps'])}. Explain these disposal steps simply in 3 sentences."),
     ("recycle",  f"{ctx} Recycling facts: {kb['recycle']} Explain the recycling options in 2-3 sentences."),
     ("impact",   f"{ctx} Decomposition: {kb['decomp']}. Impact: {kb['impact']} Educate the user about environmental impact and give one sustainability tip: {kb['tip']}"),
    ]
    fb = {"identify":f"This looks like {kb['name']}. Put it in: {kb['bin']}.","dispose":" ".join(kb['steps']),
          "recycle":kb['recycle'],"impact":f"{kb['impact']} Tip: {kb['tip']}"}
    for sec,q in jobs:
        try: txt = run(q); src = "llama-executorch" if len(txt)>20 else "kb"
        except Exception as e: txt, src = "", "kb"
        yield sec, (txt if len(txt)>20 else fb[sec]), src

# ---------- FAST PATH: ONE model load per request, capped tokens, live streaming (ON-DEVICE ONLY) ----------
import time
_cache = {}
MAXNEW = int(os.environ.get("LLAMA_MAX_NEW", "40"))
ECHO = os.environ.get("LLAMA_ECHO", "1") == "1"   # runner prints the prompt back? set 0 if not
def _prompt(desc, kb, lang="en"):
    # The 128-token PTE has limited room; the UI adds complete verified facts separately.
    return (f"Item:{desc[:32]};type:{kb['name']};bin:{kb['bin']}. "
            "Give one safe, complete disposal recommendation in one concise sentence." + LANGS.get(lang, ""))
LANGS = {
    "hi": " Reply in Hindi using Devanagari script.",
    "te": " Reply in Telugu using Telugu script.",
    "kn": " Reply in Kannada using Kannada script.",
    "mr": " Reply in Marathi using Devanagari script.",
}
def _vis(buf): return re.sub(r"<\|[^>]*\|>", "", buf.split("PyTorchObserver")[0])
def _gen(user_prompt, key, max_new, required_sections=()):
    """yields ('delta',text) while generating, then ('final',text,source). Output always ends on a full sentence."""
    if key in _cache and not _is_irrelevant_refusal(_cache[key]): yield ("final", _cache[key], "cache"); return
    prompt = _tmpl(user_prompt)
    # llama_main supports max_new_tokens and caps it against the PTE context.
    # Avoid estimating tokens from characters, which can cut answers early.
    budget = max(32, int(max_new))
    cmd = [RUNNER,"--model_path",MODEL,"--tokenizer_path",TOK,"--prompt",prompt,"--max_new_tokens",str(budget),"--temperature",TEMP]
    t0, buf, started, sent = time.time(), "", not ECHO, 0
    chunks = queue.Queue()
    with _lock:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        def read_output():
            try:
                while True:
                    data = os.read(p.stdout.fileno(), 512)
                    if not data: break
                    chunks.put(data)
            finally:
                chunks.put(None)
        reader = threading.Thread(target=read_output, daemon=True); reader.start()
        try:
            ended = False
            while not ended and time.time() - t0 < TIMEOUT:
                try: chunk = chunks.get(timeout=min(.25, max(.01, TIMEOUT - (time.time() - t0))))
                except queue.Empty: continue
                if chunk is None: ended = True; continue
                buf += chunk.decode("utf-8", "ignore")
                if not started:
                    if prompt in buf: buf = buf.split(prompt, 1)[1]; started = True
                    else: continue
                v = _vis(buf)
                # Hold the opening fragment until it is long enough to distinguish a
                # useful answer from the model's occasional generic safety refusal.
                if _is_irrelevant_refusal(v): sent = len(v); continue
                if len(v) < 48 and not _complete(v): continue
                if len(v) > sent: yield ("delta", v[sent:]); sent = len(v)
        finally:
            if p.poll() is None: p.kill()
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
    if not started:   # echo text differed slightly: keep only what follows the assistant header
        buf = buf.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    text = _trim(_vis(buf).strip())
    # Never replace the grounded disposal facts in the UI with a clipped fragment.
    # When the existing 128-token PTE stops mid-sentence, let app.py use its KB answer.
    if _is_irrelevant_refusal(text) or not _complete(text): text = ""
    if required_sections:
        if not all(re.search(re.escape(label), text, re.I) for label in required_sections):
            text = ""
        else:
            text = re.sub(r"\s*(?=(?:Reasoning|Learn):)", "\n", text, flags=re.I)
    if len(text) > 20: _cache[key] = text
    yield ("final", text if len(text) > 20 else "", "llama-executorch")
def stream(desc, cat, kb, lang="en"):
    yield from _gen(_prompt(desc, kb, lang), (cat, desc.split("(")[0], lang), MAXNEW)
def ask(question, cat, kb, lang="en"):
    """Free-form education Q&A, grounded in KB facts when the question mentions a known material."""
    ctx = (f"Item:{kb['name']}; bin:{kb['bin']}; step:{kb['steps'][0]}. " if kb else "")
    q = (f"{ctx}Q:{question[:70]}. Give one safe disposal recommendation in one complete sentence, max 16 words."
         + LANGS.get(lang, ""))
    # The app adds grounded reasoning and learning facts after this short Llama advice line.
    yield from _gen(q, ("ask", question.lower()[:120], lang), MAXNEW)
stream_auto = stream
