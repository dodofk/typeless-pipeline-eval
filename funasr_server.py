#!/usr/bin/env python3
"""OpenAI-compatible /v1/audio/transcriptions shim for Fun-ASR (llama.cpp GGUF).

OpenWhispr 的 self-hosted 模式只會 POST multipart 到 /v1/audio/transcriptions，
所以這裡把它翻譯成一次 llama-funasr-cli 呼叫。

`model` 欄位在這裡是真的有用的 —— 它選量化等級：
    funasr-q4km / funasr-q5km / funasr-q8_0   (預設 q8_0)
whisper.cpp 會忽略這個欄位，這裡不會，所以階梯可以直接從 app UI 切。
"""
import atexit, cgi, json, os, re, shutil, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(ROOT, "fun-asr/runtime/llama.cpp/build/bin/llama-funasr-cli")
ENC = os.path.join(ROOT, "models/funasr-gguf/funasr-encoder-f16.gguf")
LLM = {"q4km": "qwen3-0.6b-q4km.gguf", "q5km": "qwen3-0.6b-q5km.gguf", "q8_0": "qwen3-0.6b-q8_0.gguf"}
VAD = os.path.join(ROOT, "models/funasr-gguf/fsmn-vad.gguf")
PORT = int(os.environ.get("FUNASR_PORT", "8379"))
USE_VAD = os.environ.get("FUNASR_VAD", "1") == "1"
# OpenWhispr 會把 custom dictionary 當 OpenAI 的 `prompt` 欄位送過來
# (audioManager.js:3112)。Fun-ASR 的 decoder 是 Qwen3，prompt 是真的指令槽，
# 所以把字典包成術語表塞進去，app 端不用改就生效。
# 實測 (long3_action, q8_0)：無 prompt 14.1% CER / 12-14 術語
#                            +術語表   3.4% CER / 13-14 術語，`push` 救回來。
# 繁體救不回來 —— 模型本身只輸出簡體，那一段要靠下游 LLM 潤稿。
PROMPT_TMPL = os.environ.get(
    "FUNASR_PROMPT_TMPL", "語音轉寫。可能出現的專有名詞：{dict}。")
LOCK = threading.Lock()   # 16GB 機器，一次只跑一個


class Worker:
    """常駐的 llama-funasr-cli --serve。

    模型載入約 1.2s，一次性模式每個 request 都要付；常駐之後只付一次，
    短句延遲從 ~1.4-2.9s 掉到 ~0.3s。

    只養一個 worker：切量化就換人（重付一次 1.2s）。16GB 機器不同時扛三顆。
    """

    def __init__(self):
        self.proc = None
        self.quant = None

    def _spawn(self, quant):
        cmd = [BIN, "--enc", ENC, "-m", os.path.join(ROOT, "models/funasr-gguf", LLM[quant]), "--serve"]
        if USE_VAD:
            cmd += ["--vad", VAD]
        t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        if (self.proc.stdout.readline() or "").strip() != "__ready__":
            raise RuntimeError(f"worker {quant} failed to start")
        self.quant = quant
        print(f"[funasr] worker {quant} resident in {time.monotonic()-t0:.2f}s", flush=True)

    def _kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("__quit__\n"); self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc, self.quant = None, None

    def _ask(self, wav, prompt=""):
        # serve 協定：一行一個 request，「path」或「path\tprompt」
        line = wav if not prompt else wav + "\t" + prompt.replace("\t", " ").replace("\n", " ")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("worker died mid-request")
            if line.strip() == "__end__":
                break
            out.append(line.rstrip("\n"))
        return "".join(out).strip()

    def transcribe(self, wav, quant, prompt=""):
        if self.proc is None or self.proc.poll() is not None or self.quant != quant:
            self._kill()
            self._spawn(quant)
        try:
            return self._ask(wav, prompt)
        except RuntimeError:          # worker died — respawn once and retry
            print("[funasr] worker died, respawning", flush=True)
            self._kill(); self._spawn(quant)
            return self._ask(wav, prompt)


WORKER = Worker()
atexit.register(WORKER._kill)

def pick(model_field):
    m = (model_field or "").lower()
    for k in LLM:
        if k in m.replace("-", "_"):
            return k
    return "q8_0"

def to_wav16k(src, dst):
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
                   check=True, capture_output=True)

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _cors(self):
        # renderer 跑在 localhost:5183，殼在 127.0.0.1:8379 —— 不同來源。
        # 少了這幾個 header，請求會照樣送達並處理完，但瀏覽器會擋掉回應，
        # 前端只看得到一句 "Failed to fetch"。whisper.cpp 的 server 也是這樣送的。
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": f"funasr-{k}", "object": "model", "owned_by": "local"} for k in LLM]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not re.match(r"^/(v1/)?audio/transcriptions/?$", self.path.split("?")[0]):
            return self._json(404, {"error": {"message": f"no route {self.path}"}})
        try:
            fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
                "REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]})
            item = fs["file"]
            quant = pick(fs.getvalue("model", ""))
            dictionary = (fs.getvalue("prompt", "") or "").strip()
            prompt = PROMPT_TMPL.format(dict=dictionary) if dictionary else ""
            d = tempfile.mkdtemp(prefix="funasr-")
            try:
                raw = os.path.join(d, "in" + (os.path.splitext(item.filename or "")[1] or ".wav"))
                with open(raw, "wb") as f:
                    shutil.copyfileobj(item.file, f)
                wav = os.path.join(d, "in16k.wav")
                to_wav16k(raw, wav)
                t0 = time.monotonic()
                with LOCK:
                    text = WORKER.transcribe(wav, quant, prompt)
                dt = time.monotonic() - t0
                print(f"[funasr] {quant} {os.path.getsize(raw)/1024:.0f}KB -> {dt:.2f}s"
                      f"{'  [dict]' if dictionary else ''}  {text[:60]}", flush=True)
                self._json(200, {"text": text})
            finally:
                shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            print(f"[funasr] ERROR {e}", flush=True)
            self._json(500, {"error": {"message": str(e)}})

if __name__ == "__main__":
    for p in (BIN, ENC, VAD):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    print(f"[funasr] :{PORT}/v1/audio/transcriptions  vad={USE_VAD}  quants={list(LLM)}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
