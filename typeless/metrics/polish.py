"""潤稿層 metrics。

分兩類:
  有 reference(evalset 的 <id>-tw.txt)—— CER、術語保留、長度比 vs 正解
  無 reference —— 語助詞殘留、口吃殘留、簡體殘留、長度比 vs 輸入、幻覺

無參考的那組永遠算得出來,所以 legacy-real(沒有正解)也能評分。
"""

from __future__ import annotations

from ..lexicon import count_filler_a, count_filler_b, count_stutter
from ..norm import canon, cer, script_of, simplified_chars
from . import halluc
from .asr import length_ratio, term_recall


def _delta(before: int, after: int) -> dict:
    """一組「輸入有幾個 → 輸出剩幾個」,順帶算移除率。

    移除率的分母是輸入 —— 輸入本來就沒有的話,removed 是 None 不是 1.0。
    坑 #3 的教訓:v1 移除 0 個、v2 移除 27%,這個比率就是拿來分辨兩者的。"""
    return {"in": before, "out": after,
            "removed": round(1 - after / before, 4) if before else None}


def score(item, out: str, timings: dict | None = None,
          n_cjk: int = halluc.N_CJK, n_latin: int = halluc.N_LATIN) -> dict:
    """item 是 tl.dataset.Item(要有 .raw),out 是模型潤完的文字。"""
    timings = timings or {}

    r = {
        # ---- 無參考:清理得乾不乾淨
        "filler_a": _delta(count_filler_a(item.raw), count_filler_a(out)),
        "filler_b": _delta(count_filler_b(item.raw), count_filler_b(out)),
        "stutter": _delta(count_stutter(item.raw), count_stutter(out)),
        "simp": _delta(len(simplified_chars(item.raw)), len(simplified_chars(out))),
        "script": script_of(out),

        # ---- 無參考:有沒有刪過頭(坑 #5)
        "len_ratio_raw": length_ratio(item.raw, out),

        # ---- 無參考:有沒有掰東西(坑 #6)
        "halluc": halluc.novel_spans(item.raw, out, n_cjk, n_latin).as_dict(),

        # ---- 速度。tok/s 只有在獨佔機器時才可比(坑 #7),由 registry 標記。
        "latency_s": round(timings["wall_s"], 3) if "wall_s" in timings else None,
        "gen_tok": timings.get("gen_tok"),
        "tok_s": round(timings["tok_s"], 2) if timings.get("tok_s") else None,
        "prompt_tok": timings.get("prompt_tok"),
    }

    # ---- 有 reference 才算得出來的
    if item.ref:
        r["cer_ref"] = round(cer(item.ref, out), 4)
        r["len_ratio_ref"] = length_ratio(item.ref, out)
        r["term_keep"] = term_recall(item.terms, out)
        # 正解本身應該是乾淨的。如果 ref 的 filler_a > 0,是 ref 的品質問題,要知道。
        r["ref_filler_a"] = count_filler_a(item.ref)
    else:
        r["cer_ref"] = r["len_ratio_ref"] = r["term_keep"] = r["ref_filler_a"] = None

    return r
