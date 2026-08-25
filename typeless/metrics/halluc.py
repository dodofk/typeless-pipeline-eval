"""幻覺 / 內容插入 —— 潤稿層的 CER(坑 #6)。

問題:Qwen 把輸入的 `T S mini` 掰成 "TypeScript mini",把兩個分開的子句
「其實世界上更」+「這邊是有做壓縮過一次的」融成一句「其實世界上有更壓縮過一次的」。
這些以前只有人讀才抓得到。

做法:**新 n-gram 偵測**。輸出裡每一段字,如果它的 n-gram 在輸入裡找不到,
就是「推導不出輸入」的新內容。

正規化先過 canon(),所以下面這些**不算**幻覺:
  繁簡轉換、標點補正、大小寫、中文數字轉阿拉伯、術語斷字(top k → top-K)。
這是刻意的 —— 那些正是潤稿層該做的事。

n 的選擇:
  CJK n=3。n=2 太吵(中文任兩字的組合太容易撞),n=4 太鬆(整句改寫才抓得到)。
  Latin n=4。英文每字元熵較低,要多一個字元才有鑑別度。
  兩個都可調,且會寫進 run record —— 換了 n 舊分數就不可比,必須留痕。

**來源變體**:光比對原始輸入會有偽陽性 —— 模型照 prompt 刪掉語助詞、收合口吃
之後,兩端會接出輸入裡沒有的新相鄰字(實測:輸入「減減少大概兩成」→ 輸出
「減少兩成」,接縫「少兩成」被誤標為幻覺)。這些刪除是 prompt 要求的合法行為,
不該算幻覺。所以來源索引同時包含原文、收合口吃後、以及再去掉語助詞後三種變體,
任一命中就算有依據。定義因此是:

    幻覺 = 連「照 prompt 該刪的都刪掉」之後也推導不出來的內容。

**有界子序列**:去冗會刪掉語助詞以外的字(輸入「檢查的項目」→ 輸出「檢查項目」),
一樣會接出新相鄰。所以視窗不要求在輸入裡連續出現,只要求**依序出現且中間跳過的
字元數不超過 SLACK**。有界很重要 —— 無界的話任何字串都能在夠長的輸入裡湊出來
(實測:「有更壓」在 clip-18 的輸入裡確實依序存在,但跨了半句話,那是真的融接)。

這是**偵測不是判決**。rate 只是排序訊號,真正要看的是 spans 清單。
語意漂移(字都在輸入裡、但意思被重組掉)這一層抓不到,那是 tl/judge.py 的工作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..lexicon import collapse_stutter, drop_filler_a, drop_filler_b
from ..norm import canon

# 把 canon 過的字串切成同質片段:一段中日韓、一段英文、一段數字。
SEG = re.compile(r"[一-鿿]+|[a-z]+|[0-9]+(?:\.[0-9]+)?|%")
_CJK_SEG = re.compile(r"^[一-鿿]+$")

N_CJK = 3
N_LATIN = 4
SLACK = 2       # 視窗內允許輸入被刪掉幾個字元


@dataclass
class HallucResult:
    rate: float                                  # 被標記字元 / 輸出總字元
    novel_chars: int
    total_chars: int
    spans: list[str] = field(default_factory=list)     # 被標記的片段,人工複核用

    def as_dict(self) -> dict:
        return {"rate": round(self.rate, 4), "novel_chars": self.novel_chars,
                "total_chars": self.total_chars, "spans": self.spans}


def source_variants(src: str) -> list[str]:
    """輸入的合法變體。任一命中就算「推導得出來」。"""
    base = canon(src)
    collapsed = collapse_stutter(base)
    stripped = collapse_stutter(drop_filler_b(drop_filler_a(base)))
    return list(dict.fromkeys([base, collapsed, stripped]))


def _subseq_within(window: str, src: str, slack: int) -> bool:
    """window 是否為 src 的子序列,且整段跨度 <= len(window) + slack。"""
    limit = len(window) + slack
    start = src.find(window[0])
    while start != -1:
        i, k = start + 1, 1
        while k < len(window) and i < len(src) and i - start < limit:
            if src[i] == window[k]:
                k += 1
            i += 1
        if k == len(window):
            return True
        start = src.find(window[0], start + 1)
    return False


def _in_any(window: str, variants: list[str], slack: int = SLACK) -> bool:
    if any(window in v for v in variants):        # 連續出現,最常見,先走這條
        return True
    return any(_subseq_within(window, v, slack) for v in variants)


def _novel_mask(seg: str, src: list[str], n: int) -> list[bool]:
    """seg 的每個字元:它所屬的每個 n-gram 視窗都在 src 裡找不到 → True。

    「每個視窗都找不到」而不是「任一視窗找不到」是刻意的 —— 前者只標記真正
    孤立的新內容,後者會把每個合法片段的頭尾都標起來(邊界偽陽性)。
    """
    if len(seg) < n:                       # 太短,整段當一個單位
        return [not _in_any(seg, src)] * len(seg)

    mask = [True] * len(seg)
    for i in range(len(seg) - n + 1):
        if _in_any(seg[i:i + n], src):     # 這個視窗有依據 → 覆蓋到的字都洗白
            for j in range(i, i + n):
                mask[j] = False
    return mask


def novel_spans(src: str, out: str, n_cjk: int | None = None,
                n_latin: int | None = None) -> HallucResult:
    """out 相對於 src 的新內容。src 是**輸入逐字稿**,不是 reference ——
    幻覺的定義是「輸入裡沒有的」,不是「跟正解不一樣的」。"""
    n_cjk = N_CJK if n_cjk is None else n_cjk
    n_latin = N_LATIN if n_latin is None else n_latin
    src_v = source_variants(src)
    out_c = canon(out)
    if not out_c:
        return HallucResult(0.0, 0, 0, [])

    novel_total, spans = 0, []
    for m in SEG.finditer(out_c):
        seg = m.group(0)
        n = n_cjk if _CJK_SEG.match(seg) else n_latin
        mask = _novel_mask(seg, src_v, n)
        novel_total += sum(mask)
        # 把連續的 True 收成片段
        i = 0
        while i < len(seg):
            if mask[i]:
                j = i
                while j < len(seg) and mask[j]:
                    j += 1
                spans.append(seg[i:j])
                i = j
            else:
                i += 1

    total = sum(len(m.group(0)) for m in SEG.finditer(out_c))
    return HallucResult(novel_total / max(total, 1), novel_total, total, spans)


def score(item_raw: str, out: str, **kw) -> dict:
    return novel_spans(item_raw, out, **kw).as_dict()
