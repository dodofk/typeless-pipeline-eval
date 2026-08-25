"""eval 資料集 —— 定義「要評什麼」,不負責產生資料。

ASR 那一層由外部(OpenWhispr 或其他 code)產出,這裡只吃它的副產物。

evalset/ 的格式(權威定義):

    evalset/
      audio/<id>.wav        原始錄音。這裡不碰,留給重跑 ASR 用。
      text/<id>-raw.txt     ASR 逐字稿  = 潤稿層的 input
      text/<id>-tw.txt      人工潤過的乾淨文字 = 潤稿層的 reference
      text/<id>-asr.txt     (選用)人工逐字稿 = ASR 層的 reference。
                            有這個才算得出 ASR CER;沒有就只有潤稿層分數。
      meta.jsonl            (選用)一行一 item,補 dur_s / terms / tags。
                            沒有就從檔名推,dur_s 用 ffprobe 讀 audio/。

meta.jsonl 一行長這樣:
    {"id":"18","dur_s":76.4,"terms":[["skill"],["typescript","ts"]],
     "asr":{"source":"openwhispr","model":"funasr-q8_0"},"tags":["real","meeting"]}
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass
class Item:
    """一個 eval 單位。"""
    id: str
    raw: str = ""                       # 潤稿層 input(ASR 逐字稿)
    ref: str | None = None              # 潤稿層 reference(人工潤過的)
    asr_ref: str | None = None          # ASR 層 reference(人工逐字稿),通常沒有
    dur_s: float | None = None          # 音檔秒數,算 RTF 用
    terms: list[tuple[str, ...]] = field(default_factory=list)   # 術語表,每項是等價寫法
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    items: list[Item]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def get(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)


# ---------------------------------------------------------------- evalset
def _ffprobe_seconds(wav: pathlib.Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
            capture_output=True, text=True, timeout=30)
        return round(float(out.stdout.strip()), 2)
    except Exception:
        return None


def load_evalset(root: pathlib.Path | str = None) -> Dataset:
    """讀 evalset/。缺 meta.jsonl 也能跑 —— 從檔名推。"""
    root = pathlib.Path(root) if root else ROOT / "evalset"
    if not root.exists():
        raise SystemExit(
            f"找不到 {root}/。格式見 tl/dataset.py 的 docstring:\n"
            f"  {root}/text/<id>-raw.txt   ASR 逐字稿(input)\n"
            f"  {root}/text/<id>-tw.txt    人工潤過的正解(reference)")

    text = root / "text"
    metas: dict[str, dict] = {}
    mpath = root / "meta.jsonl"
    if mpath.exists():
        for line in mpath.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                metas[str(m["id"])] = m

    ids = sorted({p.name.rsplit("-", 1)[0] for p in text.glob("*-raw.txt")}
                 | set(metas)) if text.exists() else sorted(metas)

    items = []
    for i in ids:
        m = metas.get(i, {})

        def read(suffix):
            p = text / f"{i}-{suffix}.txt"
            return p.read_text().strip() if p.exists() else None

        wav = root / "audio" / f"{i}.wav"
        items.append(Item(
            id=i,
            raw=read("raw") or "",
            ref=read("tw"),
            asr_ref=read("asr"),
            dur_s=m.get("dur_s") or (_ffprobe_seconds(wav) if wav.exists() else None),
            terms=[tuple(t) if isinstance(t, (list, tuple)) else (t,)
                   for t in m.get("terms", [])],
            tags=m.get("tags", []),
            meta={k: v for k, v in m.items()
                  if k not in ("id", "dur_s", "terms", "tags")},
        ))
    return Dataset(root.name, items)


# ---------------------------------------------------------------- 舊資料集
# 這兩個是為了「新 pipeline 要能重現舊腳本的數字」而存在的(brief §4)。
# evalset 上線之後可以收掉,但先留著當回歸基準。

# 從 score.py 的 K 表搬過來,一字未改。
_LEGACY_TERMS = {
    "long1_meeting": [("baseline",), ("model",), ("accuracy",), ("latency",), ("request",),
                      ("production",), ("batch size", "batchsize"), ("quantization",), ("int8",),
                      ("trade-off", "tradeoff", "trade off"), ("七十八點五", "78.5"),
                      ("一點二", "1.2"), ("三十二", "32"), ("六十四", "64")],
    "long2_technical": [("react",), ("typescript",), ("fastapi",), ("postgresql",), ("redis",),
                        ("a100",), ("vllm",), ("inference",), ("throughput",), ("token",),
                        ("bge-m3", "bgem3"), ("qdrant",), ("top-k", "topk"), ("hybrid search",),
                        ("bm25",), ("dense",), ("rerank",), ("六十", "60"), ("一零二四", "1024"),
                        ("零點三", "0.3"), ("零點七", "0.7")],
    "long3_action": [("action item",), ("小陳",), ("evaluation",), ("script",), ("push",),
                     ("repo",), ("amy",), ("slack",), ("八月二十二", "8月22"),
                     ("八月二十五", "8月25"), ("八月二十七", "8月27"), ("八月二十八", "8月28"),
                     ("兩點半", "2點半"), ("十點", "10點")],
}

# 從 RESULTS_20260818_tts_bakeoff.md 抄回來的音檔長度。
_LEGACY_TTS_DUR = {
    "mm28_long1_meeting_f10": 43.2, "mm28_long1_meeting_m115": 34.4, "say_long1_meeting": 50.2,
    "mm28_long2_technical_f10": 45.1, "mm28_long2_technical_m115": 44.1,
    "say_long2_technical": 51.8,
    "mm28_long3_action_f10": 37.9, "mm28_long3_action_m115": 32.1, "say_long3_action": 45.6,
}

# 從 bench_polish.py 的 DUR 表搬過來。
_LEGACY_REAL_DUR = {"14": 69.0, "16": 28.1, "17": 33.1, "18": 76.4}


def load_legacy_tts(root: pathlib.Path | str = None) -> Dataset:
    """gold/*.txt + tts/*.wav —— score.py 那張表的資料集。

    這裡的 gold 是**人工逐字稿**(ASR reference),不是潤稿 reference,
    所以填進 asr_ref 而不是 ref。"""
    root = pathlib.Path(root) if root else ROOT
    items = []
    for stem, terms in _LEGACY_TERMS.items():
        gold = (root / "gold" / f"{stem}.txt").read_text().strip()
        for tag in (f"mm28_{stem}_f10", f"mm28_{stem}_m115", f"say_{stem}"):
            items.append(Item(id=tag, asr_ref=gold, dur_s=_LEGACY_TTS_DUR.get(tag),
                              terms=terms, tags=["tts", stem]))
    return Dataset("legacy-tts", items)


def load_legacy_real(root: pathlib.Path | str = None) -> Dataset:
    """out/e2e_{14,16,17,18}_raw.txt —— bench_polish.py 那張表的資料集。

    沒有任何 reference(這正是 evalset 要補的洞),所以只能跑無參考的潤稿 metric。"""
    root = pathlib.Path(root) if root else ROOT
    items = []
    for k in ("16", "17", "14", "18"):
        p = root / "out" / f"e2e_{k}_raw.txt"
        if p.exists():
            items.append(Item(id=k, raw=p.read_text().strip(),
                              dur_s=_LEGACY_REAL_DUR.get(k), tags=["real", "zh"]))
    return Dataset("legacy-real", items)


def load_legacy_real_en(root: pathlib.Path | str = None) -> Dataset:
    """out/en_{14,16,17,18}_raw.txt —— bench_polish_en.py 的英文對照組。"""
    root = pathlib.Path(root) if root else ROOT
    items = []
    for k in ("16", "17", "14", "18"):
        p = root / "out" / f"en_{k}_raw.txt"
        if p.exists():
            items.append(Item(id=k, raw=p.read_text().strip(),
                              dur_s=_LEGACY_REAL_DUR.get(k), tags=["real", "en"]))
    return Dataset("legacy-real-en", items)


LOADERS = {
    "evalset": load_evalset,
    "legacy-tts": load_legacy_tts,
    "legacy-real": load_legacy_real,
    "legacy-real-en": load_legacy_real_en,
}


def load(name: str) -> Dataset:
    if name in LOADERS:
        return LOADERS[name]()
    if pathlib.Path(name).exists():          # 直接給路徑 = 當 evalset 讀
        return load_evalset(name)
    raise SystemExit(f"不認識的 dataset: {name}\n可用:{', '.join(LOADERS)} 或 evalset 目錄路徑")
