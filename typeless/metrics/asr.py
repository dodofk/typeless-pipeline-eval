"""ASR 層 metrics。

這一層的模型不是我們跑的(輸入來自 OpenWhispr 或其他 code 的副產物),
但只要有人工逐字稿當 reference 就能評分。

沒有 asr_ref 的資料集只會拿到 len_ratio / script / rtf —— CER 和術語召回會是 None。
這不是 bug,是誠實回報「這個 metric 現在量不了」。
"""

from __future__ import annotations

from ..norm import canon, cer, script_of, simplified_chars


def term_recall(terms: list[tuple[str, ...]], hyp: str) -> dict | None:
    """術語 + 數字召回。terms 每一項是「等價寫法」的 tuple,任一命中就算有。

    比對走 canon(),所以「零點三 / 0.3」「BGE-M3 / BGE M3」不用另外列等價寫法 ——
    但 score.py 的舊表還是把它們列著,列了不會錯,只是多餘。"""
    if not terms:
        return None
    h = canon(hyp)
    missing = [t[0] for t in terms if not any(canon(a) in h for a in t)]
    return {"hit": len(terms) - len(missing), "total": len(terms), "missing": missing}


def length_ratio(ref: str, hyp: str) -> float | None:
    """len(hyp)/len(ref),正規化後的字元數。

    坑 #5:這是抓「靜默吞句」的護欄。whisper 在 >40s 音檔會整句消失,
    CER 只會小幅上升,但長度比會掉到 0.79 —— 那才是真正的訊號。"""
    r = canon(ref)
    return round(len(canon(hyp)) / len(r), 4) if r else None


def rtf(elapsed_s: float | None, dur_s: float | None) -> float | None:
    """即時率 real-time factor:處理秒數 / 音檔秒數。越小越快,<1 才跟得上說話。"""
    if not elapsed_s or not dur_s:
        return None
    return round(elapsed_s / dur_s, 4)


def score(item, hyp: str, elapsed_s: float | None = None) -> dict:
    """item 是 tl.dataset.Item。hyp 是 ASR 的逐字稿。"""
    ref = item.asr_ref
    return {
        "cer": round(cer(ref, hyp), 4) if ref else None,
        "term_recall": term_recall(item.terms, hyp),
        "len_ratio": length_ratio(ref, hyp) if ref else None,
        "rtf": rtf(elapsed_s, item.dur_s),
        "script": script_of(hyp),
        "simp_chars": len(simplified_chars(hyp)),
        "n_chars": len(canon(hyp)),
    }
