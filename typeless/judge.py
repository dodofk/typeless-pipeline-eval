"""LLM-as-a-judge:語意漂移(坑 #6 的第二層)。

halluc.py 抓的是**字面新內容** —— 輸入裡找不到的字。它抓不到另一種:
字都在輸入裡,但被重組成一個輸入沒有的主張。這一層交給 judge。

原則:
  - judge 是**雲端模型**。本機 judge 會跟被測模型搶記憶體頻寬(坑 #7),
    而且被測的就是本機模型的判斷力 —— 拿它當裁判是循環論證。
  - temp=0 + 固定 seed。judge 不決定性的話,分數就是雜訊(坑 #3 同理)。
  - judge model 與 judge prompt 的 sha 都寫進 run record。換了裁判,舊分數不可比。
  - 輸出**結構化**,不要自由文字 —— 自由文字沒辦法累積、沒辦法比較。

MiniMax-M3 是 reasoning model:reasoning_content 會吃掉 max_tokens 預算,
所以 max_tokens 要給夠(預設 4096),而且答案只看 content 不看 reasoning_content。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

from .engines import sha8

MINIMAX_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"

JUDGE_PROMPT = """You are auditing a speech-transcript cleanup system for MEANING DRIFT.

You get INPUT (a raw ASR transcript, Taiwanese Mandarin with embedded English, \
full of disfluency) and OUTPUT (the cleaned version a model produced).

The cleanup model is ALLOWED to: delete fillers and stutters, fix punctuation, \
convert Simplified to Traditional Chinese, fix casing and spacing of English terms, \
repair obvious homophone errors, and convert spoken numbers to digits.

The cleanup model is NOT allowed to: assert anything the speaker did not assert, \
merge two separate statements into one new claim, change who did what to whom, \
change a negation into an affirmation (or vice versa), invent names, numbers, or \
technical terms, or delete a statement the speaker actually made.

Report ONLY meaning drift. Do not report style, wording, tone, or formatting.
When in doubt, do NOT report it — a false alarm is worse than a miss here.

Respond with JSON only, no prose, no markdown fence:
{"drifts":[{"output_span":"<verbatim text from OUTPUT>",\
"input_basis":"<the input text it should have come from, or \\"\\" if none>",\
"kind":"invented|merged|negated|reassigned|dropped",\
"severity":"high|low",\
"why":"<one short sentence>"}]}

An empty list means no meaning drift. That is a normal and common answer."""

USER_TEMPLATE = "INPUT:\n{raw}\n\nOUTPUT:\n{out}"


@dataclass
class JudgeConfig:
    model: str = "MiniMax-M3"
    url: str = MINIMAX_URL
    api_key_env: str = "MINIMAX_API_KEY"
    temp: float = 0.0
    max_tokens: int = 4096
    timeout: int = 300
    retries: int = 2
    prompt_sha: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["prompt_sha"] = d["prompt_sha"] or sha8(JUDGE_PROMPT)
        return d


@dataclass
class JudgeResult:
    drifts: list[dict] = field(default_factory=list)
    error: str | None = None
    raw_reply: str = ""
    usage: dict = field(default_factory=dict)

    @property
    def high(self) -> int:
        return sum(1 for d in self.drifts if d.get("severity") == "high")

    def as_dict(self) -> dict:
        return {"n_drift": len(self.drifts), "n_high": self.high,
                "drifts": self.drifts, "error": self.error}


def _extract_json(s: str) -> dict | None:
    """模型有時候還是會包 ```json 或加一句廢話。抓第一個平衡的 {...}。"""
    s = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s.strip())
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def judge(raw: str, out: str, cfg: JudgeConfig | None = None) -> JudgeResult:
    cfg = cfg or JudgeConfig()
    key = os.environ.get(cfg.api_key_env)
    if not key:
        return JudgeResult(error=f"{cfg.api_key_env} 沒設 —— 跳過語意漂移判定")

    body = json.dumps({
        "model": cfg.model,
        "messages": [{"role": "system", "content": JUDGE_PROMPT},
                     {"role": "user", "content": USER_TEMPLATE.format(raw=raw, out=out)}],
        "temperature": cfg.temp, "max_tokens": cfg.max_tokens,
    }).encode()

    last = None
    for attempt in range(cfg.retries + 1):
        try:
            req = urllib.request.Request(
                cfg.url, body,
                {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            d = json.load(urllib.request.urlopen(req, timeout=cfg.timeout))
            msg = d["choices"][0]["message"]
            reply = (msg.get("content") or "").strip()
            parsed = _extract_json(reply)
            if parsed is None:
                # content 空通常是 reasoning 把 max_tokens 吃光了
                last = ("回覆不是 JSON(可能 max_tokens 被 reasoning 吃光)"
                        if not reply else f"回覆不是 JSON: {reply[:160]}")
                time.sleep(1 + attempt)
                continue
            drifts = parsed.get("drifts") or []
            if not isinstance(drifts, list):
                drifts = []
            return JudgeResult(drifts=drifts, raw_reply=reply, usage=d.get("usage", {}))
        except Exception as e:
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                detail = ": " + e.read().decode("utf-8", "replace")[:200]
            last = f"{type(e).__name__}{detail}"
            time.sleep(1 + attempt)
    return JudgeResult(error=last)
