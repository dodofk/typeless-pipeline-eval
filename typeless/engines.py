"""外部服務的 adapter。

這個 codebase **不啟動模型** —— llama-server / STT server 由使用者自己起。
這裡只負責:把文字送過去、把結果和 timing 收回來、把環境的可信度記下來。

ASR 那一層更是完全不碰:輸入是 OpenWhispr(或其他 code)已經產好的逐字稿,
從 evalset/text/<id>-raw.txt 讀進來。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent

# gemma / qwen 系列會在前面掛 <think>...</think>。潤稿不需要 reasoning trace,
# request 裡已經關了,這裡是第二道保險。
THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


# ---------------------------------------------------------------- 環境
def llama_server_count() -> int:
    """機器上有幾個 llama-server 在跑。

    坑 #7:這是 16GB 機器,模型是記憶體頻寬綁定的。同時起兩個會互搶頻寬,
    tok/s 就不可比。這個數字會寫進 run record —— >1 的 run,速度數字不可信。"""
    try:
        out = subprocess.run(["pgrep", "-f", "llama-server"],
                             capture_output=True, text=True, timeout=10)
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return -1                       # 測不到,誠實回報 -1 而不是假裝是 1


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def load_prompt(path: str | pathlib.Path, agent_name: str = "assistant") -> tuple[str, str]:
    """回傳 (prompt 內容, sha8)。sha 會進 run record ——
    prompt 檔改了但檔名沒改的話,只有 sha 認得出來。"""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / p
    text = p.read_text().replace("{{agentName}}", agent_name)
    return text, sha8(text)


# ---------------------------------------------------------------- 潤稿 LLM
@dataclass
class PolishConfig:
    url: str = "http://127.0.0.1:8080"
    model: str = "local"
    prompt_file: str = "prompts/cleanup-zhTW-mixed-v2.txt"
    temp: float = 0.0                   # 坑 #3:A/B 一定要 temp=0
    seed: int = 1234
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    enable_thinking: bool = False       # 開著的話一句話 polish 會噴 761 tok / 19.4s
    timeout: int = 1800
    user_template: str = "<transcript>\n{raw}\n</transcript>"
    prompt_sha: str = ""                # load 之後填

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolishResult:
    text: str
    timings: dict = field(default_factory=dict)
    error: str | None = None


def polish(raw: str, cfg: PolishConfig, prompt: str) -> PolishResult:
    """送一段逐字稿給 llama-server,回傳潤好的文字 + timing。"""
    body = json.dumps({
        "model": cfg.model,
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": cfg.user_template.format(raw=raw)}],
        "temperature": cfg.temp, "top_p": cfg.top_p, "top_k": cfg.top_k,
        "max_tokens": cfg.max_tokens, "seed": cfg.seed, "stream": False,
        "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
    }).encode()
    req = urllib.request.Request(f"{cfg.url}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=cfg.timeout))
    except Exception as e:
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            detail = ": " + e.read().decode("utf-8", "replace")[:300]
        return PolishResult("", {}, f"{type(e).__name__}{detail}")

    wall = time.monotonic() - t0
    txt = THINK.sub("", d["choices"][0]["message"]["content"]).strip()
    t, u = d.get("timings", {}), d.get("usage", {})
    return PolishResult(txt, {
        "wall_s": wall,
        "gen_tok": u.get("completion_tokens"),
        "prompt_tok": u.get("prompt_tokens"),
        # llama-server 自己量的 decode 速度比 wall_s 準(不含 HTTP 往返)
        "tok_s": t.get("predicted_per_second"),
        "prompt_tok_s": t.get("prompt_per_second"),
    })


def warm(cfg: PolishConfig, prompt: str) -> None:
    """暖機一次,把 system prompt 灌進 KV cache。

    不暖機的話第一個 clip 會多付一次 prompt prefill,latency 表的第一列永遠是離群值。"""
    polish("暖機。", cfg, prompt)
