"""ASR 引擎的直接呼叫 —— 產生 hypothesis 的另一條路。

為什麼要這條路:OpenWhispr 沒有 headless CLI,走 app 就得人工點。
每換一個模型就要建 folder、切 Settings、拖 14 個檔,改一次參數重來一輪。

⚠️ 但直接打引擎跟走 OpenWhispr **不一定等價**,而且哪裡不等價是查過原始碼的:

  whisper(本機)  OpenWhispr 的 transcribe-audio-file 走 "noteRecording" context,
                 VAD **預設是開的**(只有 dictation 預設關),而且 whisper-cli 的
                 VAD 預設值跟 OpenWhispr 給的不一樣:
                     參數                        whisper-cli 預設   OpenWhispr
                     --vad-min-silence-duration-ms      100            200
                     --vad-speech-pad-ms                 30            100
                     --vad-samples-overlap              0.10           0.5
                     --vad-max-speech-duration-s      無限             30
                 不複製這組值就一定對不起來。OPENWHISPR_VAD 就是那組值。
                 另外 OpenWhispr 一律加 --no-timestamps(見 whisperServer.js #1348)。

  funasr/breeze  走 self-hosted,OpenWhispr 只是把整個檔 POST 到你自己的 server,
                 沒有本機 VAD、沒有切段 → 天生等價。

  scribe(雲端)   BYOK 直送 ElevenLabs,chunking_strategy 交給對方 → 等價。
                 ⚠️ scribe 就是 ground truth 的來源引擎,它的 CER 是下界不是實力。

所以:funasr/breeze/scribe 這條路可以直接信;whisper 要拿 OpenWhispr 跑一次對照,
確認等價之後才能一直用這條路。用 ./tl asr --engine whisper --no-vad 可以量出
「VAD 到底幫了還是害了」。
"""

from __future__ import annotations

import json
import mimetypes
import os
import pathlib
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 抄自 openwhispr/src/constants/whisperVad.json 的 DEFAULTS。
OPENWHISPR_VAD = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 200,
    "max_speech_duration_s": 30,
    "speech_pad_ms": 100,
    "samples_overlap": 0.5,
}
VAD_MODEL = ROOT / "models/whisper-vad/ggml-silero-v5.1.2.bin"


@dataclass
class AsrResult:
    text: str = ""
    elapsed_s: float = 0.0
    error: str | None = None
    invocation: dict = field(default_factory=dict)   # 實際怎麼叫的,要能重現


# ---------------------------------------------------------------- multipart
def _post_audio(url: str, wav: pathlib.Path, fields: dict,
                headers: dict | None = None, timeout: int = 1800) -> str:
    b = "----tlboundary" + uuid.uuid4().hex[:12]
    parts = []
    for k, v in fields.items():
        if v is None or v == "":
            continue
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    ctype = mimetypes.guess_type(wav.name)[0] or "audio/wav"
    parts.append(
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode() + wav.read_bytes() + b"\r\n")
    body = b"".join(parts) + f"--{b}--\r\n".encode()

    req = urllib.request.Request(url, body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={b}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        return json.loads(raw).get("text", "").strip()
    except json.JSONDecodeError:
        return raw.strip()


# ---------------------------------------------------------------- whisper
def whisper(wav: pathlib.Path, model: str | None = None, vad: bool = True,
            language: str = "auto", prompt: str | None = None,
            vad_params: dict | None = None, binary: str = "whisper-cli") -> AsrResult:
    """whisper.cpp。vad=True 時複製 OpenWhispr 的那組參數(見本檔 docstring)。"""
    model = model or str(ROOT / "models/ggml-large-v3-turbo-q5_0.bin")
    vp = {**OPENWHISPR_VAD, **(vad_params or {})}
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "o"
        cmd = [binary, "-m", model, "-f", str(wav), "-l", language,
               "--no-timestamps", "-oj", "-of", str(out)]
        if prompt:
            cmd += ["--prompt", prompt]
        if vad:
            if not VAD_MODEL.exists():
                return AsrResult(error=f"缺 VAD 模型:{VAD_MODEL}\n"
                                       f"  下載:curl -L -o {VAD_MODEL} https://huggingface.co/"
                                       f"ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin")
            cmd += ["--vad", "--vad-model", str(VAD_MODEL),
                    "--vad-threshold", str(vp["threshold"]),
                    "--vad-min-speech-duration-ms", str(vp["min_speech_duration_ms"]),
                    "--vad-min-silence-duration-ms", str(vp["min_silence_duration_ms"]),
                    "--vad-max-speech-duration-s", str(vp["max_speech_duration_s"]),
                    "--vad-speech-pad-ms", str(vp["speech_pad_ms"]),
                    "--vad-samples-overlap", str(vp["samples_overlap"])]
        t0 = time.monotonic()
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        except subprocess.CalledProcessError as e:
            return AsrResult(error=f"whisper-cli 失敗:{e.stderr.decode('utf-8', 'replace')[:300]}")
        except Exception as e:
            return AsrResult(error=f"{type(e).__name__}: {e}")
        el = time.monotonic() - t0
        js = out.with_suffix(".json")
        if not js.exists():
            return AsrResult(error="whisper-cli 沒產出 json")
        data = json.loads(js.read_text())
        text = " ".join(s["text"].strip() for s in data.get("transcription", [])
                        if s.get("text", "").strip())
    return AsrResult(text.strip(), el, None,
                     {"engine": "whisper", "model": model, "vad": vad,
                      "vad_params": vp if vad else None, "language": language,
                      "prompt": prompt, "cmd": cmd[:1] + ["…"] + cmd[-2:]})


# ---------------------------------------------------------------- funasr
def funasr(wav: pathlib.Path, quant: str = "q8_0",
           url: str = "http://127.0.0.1:8379", prompt: str = "") -> AsrResult:
    """funasr_server.py 的 OpenAI 相容 shim。

    prompt 一定要跟 app 送的一致 —— gt.py 的教訓是不一致會差 8–28%,純測量誤差。"""
    t0 = time.monotonic()
    try:
        text = _post_audio(f"{url}/v1/audio/transcriptions", wav,
                           {"model": f"funasr-{quant}", "response_format": "json",
                            "prompt": prompt})
    except Exception as e:
        detail = e.read().decode("utf-8", "replace")[:200] if isinstance(
            e, urllib.error.HTTPError) else ""
        return AsrResult(error=f"{type(e).__name__}{detail}")
    return AsrResult(text, time.monotonic() - t0, None,
                     {"engine": "funasr", "quant": quant, "url": url, "prompt": prompt})


# ---------------------------------------------------------------- breeze
def breeze(wav: pathlib.Path, url: str = "http://127.0.0.1:8380",
           prompt: str = "") -> AsrResult:
    t0 = time.monotonic()
    try:
        text = _post_audio(f"{url}/v1/audio/transcriptions", wav,
                           {"model": "breeze-asr-25", "response_format": "json",
                            "prompt": prompt})
    except Exception as e:
        detail = e.read().decode("utf-8", "replace")[:200] if isinstance(
            e, urllib.error.HTTPError) else ""
        return AsrResult(error=f"{type(e).__name__}{detail}")
    return AsrResult(text, time.monotonic() - t0, None,
                     {"engine": "breeze", "url": url, "prompt": prompt})


# ---------------------------------------------------------------- scribe
def scribe(wav: pathlib.Path, model: str = "scribe_v1") -> AsrResult:
    """ElevenLabs Scribe。欄位叫 model_id 不是 model,認證是 xi-api-key 不是 Bearer。

    ⚠️ 這就是 ground truth 的來源引擎。它的 CER 量到的是「人工當初改了多少」,
    是下界不是實力,不能跟其他引擎並排比。"""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return AsrResult(error="ELEVENLABS_API_KEY 沒設")
    t0 = time.monotonic()
    try:
        text = _post_audio("https://api.elevenlabs.io/v1/speech-to-text", wav,
                           {"model_id": model, "tag_audio_events": "false"},
                           {"xi-api-key": key})
    except Exception as e:
        detail = e.read().decode("utf-8", "replace")[:200] if isinstance(
            e, urllib.error.HTTPError) else ""
        return AsrResult(error=f"{type(e).__name__}{detail}")
    return AsrResult(text, time.monotonic() - t0, None,
                     {"engine": "scribe", "model": model,
                      "warning": "GT 的來源引擎,CER 是下界不是實力"})


ENGINES = {"whisper": whisper, "funasr": funasr, "breeze": breeze, "scribe": scribe}


def run(spec: str, wav: pathlib.Path, **kw) -> AsrResult:
    """spec:whisper / whisper:no-vad / funasr:q4km / breeze / scribe"""
    name, _, arg = spec.partition(":")
    fn = ENGINES.get(name)
    if not fn:
        return AsrResult(error=f"不認識的引擎:{name},可用:{', '.join(ENGINES)}")
    if name == "whisper":
        if arg == "no-vad":
            kw["vad"] = False
        elif arg:
            kw["model"] = arg
    elif name == "funasr" and arg:
        kw["quant"] = arg
    return fn(wav, **kw)
