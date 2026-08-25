"""報表 —— 一個指令印出跨 run 對照表。

原則:
  - 數字旁邊一定要有可信度標記。tok/s 在非獨佔環境下不可比(坑 #7),
    temp>0 的 A/B 結論是雜訊(坑 #3) —— 這兩件事直接印在表上,不要靠人記得。
  - 缺的值印 `—` 不印 0。0 和「量不了」是完全不同的意思。
"""

from __future__ import annotations

from .registry import read_index

# (欄位 key, 表頭, 寬度, 格式)
POLISH_COLS = [
    ("filler_a_out", "語助A殘", 8, "int"),
    ("filler_a_removed", "移除率", 8, "pct"),
    ("stutter_out", "口吃殘", 7, "int"),
    ("simp_out", "簡殘", 6, "int"),
    ("len_ratio_raw", "長度比", 7, "num"),
    ("cer_ref", "CER", 7, "pct"),
    ("term_keep", "術語", 7, "pct"),
    ("halluc_rate", "幻覺", 7, "pct"),
    ("drift_high", "漂移", 6, "int"),
    ("tok_s", "tok/s", 8, "num"),
]

ASR_COLS = [
    ("cer", "CER", 8, "pct"),
    ("term_recall", "術語", 8, "pct"),
    ("len_ratio", "長度比", 8, "num"),
    ("rtf", "RTF", 8, "num"),
    ("simp_chars", "簡體字", 8, "int"),
]


def fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "num":
        return f"{v:.2f}"
    return str(v)


def _w(s: str) -> int:
    """終端寬度:CJK 算兩格。不算的話中文表頭會把欄位擠歪。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _pad(s: str, width: int, right: bool = True) -> str:
    gap = max(width - _w(s), 0)
    return (" " * gap + s) if right else (s + " " * gap)


def table(rows: list[dict], arm: str = "polish") -> str:
    cols = ASR_COLS if arm == "asr" else POLISH_COLS
    out = []
    head = _pad("run", 26, False) + _pad("model", 16, False) + _pad("prompt", 10, False) \
        + _pad("in", 6, False) + _pad("T", 6) + _pad("n", 5)
    head += "".join(_pad(h, w) for _, h, w, _ in cols)
    out.append(head)
    out.append("-" * _w(head))

    for r in rows:
        agg = r.get("aggregate") or {}
        label = r.get("label") or r.get("run_id", "")
        prompt = (r.get("prompt") or "").split("/")[-1].replace("cleanup-", "").replace(".txt", "")
        temp = r.get("temp")
        # 坑 #3:temp>0 的 A/B 結論是雜訊,直接在表上標出來
        tmark = "?" if (temp is not None and temp > 0) else " "
        line = _pad(label[:25], 26, False) + _pad(str(r.get("model") or "—")[:15], 16, False) \
            + _pad(prompt[:9], 10, False) \
            + _pad(r.get("input") or "—", 6, False) \
            + _pad(f"{temp}{tmark}" if temp is not None else "—", 6) \
            + _pad(str(agg.get("n", "—")), 5)
        for key, _, w, kind in cols:
            line += _pad(fmt(agg.get(key), kind), w)
        # 坑 #7:非獨佔環境下 tok/s 不可比
        if arm != "asr" and r.get("speed_trustworthy") is False:
            line += "  ⚠速度"
        out.append(line)

    notes = []
    if any((r.get("temp") or 0) > 0 for r in rows):
        notes.append("?  = temp>0,同一組 A/B 的勝負可能純粹是雜訊(坑#3)")
    if any(r.get("speed_trustworthy") is False for r in rows):
        notes.append("⚠速度 = 跑的時候不只一個 llama-server,tok/s 不可跨 run 比(坑#7)")
    if notes:
        out.append("")
        out.extend("  " + n for n in notes)
    return "\n".join(out)


def report(arm: str | None = None, dataset: str | None = None,
           since: str | None = None, label_like: str | None = None) -> str:
    rows = read_index()
    if arm:
        rows = [r for r in rows if r.get("arm") == arm]
    if dataset:
        rows = [r for r in rows if r.get("dataset") == dataset]
    if since:
        rows = [r for r in rows if (r.get("created") or "") >= since]
    if label_like:
        rows = [r for r in rows if label_like.lower() in (r.get("label") or "").lower()]
    if not rows:
        return "沒有符合的 run。 ./tl list 看看有什麼。"

    out = []
    for a in sorted({r.get("arm", "polish") for r in rows}):
        sub = [r for r in rows if r.get("arm") == a]
        out.append(f"\n=== arm: {a}  ({len(sub)} runs) ===")
        out.append(table(sub, a))
    return "\n".join(out)
