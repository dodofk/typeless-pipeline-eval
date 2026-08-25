#!/usr/bin/env python3
"""OpenAI-compatible /v1/audio/transcriptions shim for Breeze-ASR-25 (transcribe.cpp).

Same contract as funasr_server.py so OpenWhispr's self-hosted mode can point at
either one without an app-side change. Difference: the model stays resident in
THIS process via the transcribe.cpp Python binding — no per-request subprocess,
no reload. Cold start ~1s (plus a one-time ~11s Metal shader compile), then
every request is pure inference.

    TRANSCRIBE_LIBRARY=.../libtranscribe.dylib .venv-mlx/bin/python breeze_server.py

OpenWhispr sends its custom dictionary as the OpenAI `prompt` field; Whisper
takes that as initial_prompt, which biases terminology AND output style
(a punctuated zh-TW prompt is what gets you punctuation back).
"""
import json, os, re, subprocess, sys, tempfile, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import transcribe_cpp

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("BREEZE_MODEL", os.path.join(ROOT, "models/breeze-asr/Breeze-ASR-25-Q5_K_M.gguf"))
PORT = int(os.environ.get("BREEZE_PORT", "8380"))
LANG = os.environ.get("BREEZE_LANG", "zh")
# Style-carrying default: punctuated Traditional Chinese, so the output gets punctuation.
PROMPT_TMPL = os.environ.get(
    "BREEZE_PROMPT_TMPL",
    "以下是一段會議錄音的逐字稿，使用繁體中文，含標點符號。可能出現的專有名詞：{dict}。")
LOCK = threading.Lock()   # 16GB machine: one decode at a time


class Engine:
    def __init__(self):
        self.model = transcribe_cpp.Model(MODEL)
        self.session = self.model.session().__enter__()
        print(f"[breeze] model resident: {os.path.basename(MODEL)}", flush=True)

    def run(self, pcm, prompt):
        family = transcribe_cpp.WhisperRunOptions(
            initial_prompt=prompt or None,
            condition_on_prev_tokens=True,   # carry context across 30s windows
            temperature=0.0,
        )
        return self.session.run(pcm, language=LANG, timestamps="none", family=family).text.strip()


ENGINE = None


def to_pcm(src):
    """Any container -> 16 kHz mono float32, via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        raw = f.name
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", src,
                        "-ar", "16000", "-ac", "1", "-f", "f32le", raw], check=True)
        return np.fromfile(raw, dtype=np.float32)
    finally:
        os.path.exists(raw) and os.unlink(raw)


def parse_multipart(body, boundary):
    """Minimal multipart reader — enough for OpenWhispr's shape, no cgi dependency."""
    out, files = {}, {}
    sep = b"--" + boundary
    for part in body.split(sep):
        if not part.strip(b"-\r\n"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if not _:
            continue
        h = head.decode("utf-8", "replace")
        name = re.search(r'name="([^"]*)"', h)
        if not name:
            continue
        data = data.rstrip(b"\r\n")
        fn = re.search(r'filename="([^"]*)"', h)
        (files if fn else out)[name.group(1)] = data if fn else data.decode("utf-8", "replace")
    return out, files


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in [("Access-Control-Allow-Origin", "*"),
                     ("Access-Control-Allow-Methods", "POST, GET, OPTIONS"),
                     ("Access-Control-Allow-Headers", "*")]:
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": "breeze-asr-25", "object": "model", "owned_by": "mediatek-research"}]})
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            return self._json(200, {"status": "ok"})
        self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if not re.match(r"^/(v1/)?audio/transcriptions/?$", self.path.split("?")[0]):
            return self._json(404, {"error": {"message": f"no route {self.path}"}})
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r'boundary="?([^";]+)"?', ctype)
        if not m:
            return self._json(400, {"error": {"message": "expected multipart/form-data"}})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        fields, files = parse_multipart(body, m.group(1).encode())
        if not files:
            return self._json(400, {"error": {"message": "no file part"}})

        blob = next(iter(files.values()))
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(blob)
            src = f.name
        try:
            pcm = to_pcm(src)
            words = fields.get("prompt", "").strip()
            prompt = PROMPT_TMPL.format(dict=words) if words else PROMPT_TMPL.format(dict="無")
            with LOCK:
                text = ENGINE.run(pcm, prompt)
            dur = len(pcm) / 16000
            print(f"[breeze] {dur:.1f}s audio -> {len(text)} chars", flush=True)
            if fields.get("response_format", "json") == "text":
                b = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self._json(200, {"text": text})
        except Exception as e:
            self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
        finally:
            os.path.exists(src) and os.unlink(src)


if __name__ == "__main__":
    if not os.path.exists(MODEL):
        sys.exit(f"model not found: {MODEL}")
    ENGINE = Engine()
    print(f"[breeze] :{PORT}/v1/audio/transcriptions  lang={LANG}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
