"""語助詞 / 口吃的定義 —— polish metric 和 halluc 共用一份。

分 tier 是照 prompts/cleanup-zhTW-mixed-v2.txt 的規則來的:

  TIER A — 純發聲,無條件刪除,沒有例外。輸出裡還有這些 = 沒清乾淨。
  TIER B — 有時是填充、有時載有意義(那個 report / 就是 BM25 加 dense)。
           殘留不算錯,所以只計數不判分。

坑 #4:計數之前一定要先去標點。曾經誤報「移除 64% 語助詞」,
實際上模型把 `對對對對` 變成 `對,對,對,對` —— 逗號插進去,重複一個都沒少。

中英要用**不同的正規化視角**,這裡自己處理,呼叫端一律傳原文:
  中文 → canon()        無空白。這樣「對,對,對」才會塌成「對對對」被數到。
  英文 → canon_spaced() 保留詞界。canon() 會把 "um, I think" 壓成 "umithink",
                        `\bum\b` 就永遠不匹配 —— 英文那組指標會全部假裝是 0。
"""

from __future__ import annotations

import re

from .norm import canon, canon_spaced

# ---------------------------------------------------------------- tier A
# 中文的純發聲。「啊」只在句中算 filler(句尾的「好啊」是語氣,不是雜訊),
# 但去標點之後沒有句界可判,所以一律計入 —— 寧可高估,兩個 run 之間是可比的。
FILLER_A_ZH = ["呃", "嗯", "欸", "齁", "啊"]
FILLER_A_EN = re.compile(r"\b(?:um+|uh+|er|erm|mmm+|hmm+|ah)\b", re.I)

# ---------------------------------------------------------------- tier B
FILLER_B_ZH = ["那個", "就是說", "就是", "然後", "基本上", "對啊", "對吧", "嘛"]
FILLER_B_EN = re.compile(r"\b(?:like|you know|i mean|sort of|kind of)\b", re.I)

# ---------------------------------------------------------------- 口吃
# 中文:同一個字或同一組兩字連續重複(對對對對、那個那個、我我我)。
STUTTER_ZH_1 = re.compile(r"([一-鿿])\1+")
STUTTER_ZH_2 = re.compile(r"([一-鿿]{2})\1+")
# 英文:同一個詞連續重複,中間可以隔逗號空白。
STUTTER_EN = re.compile(r"\b(\w+)\b(?:[,\s]+\b\1\b)+", re.I)


def count_filler_a(text: str) -> int:
    """tier A 出現次數。傳原文進來,正規化由這裡負責。"""
    return (sum(canon(text).count(f) for f in FILLER_A_ZH)
            + len(FILLER_A_EN.findall(canon_spaced(text))))


def count_filler_b(text: str) -> int:
    return (sum(canon(text).count(f) for f in FILLER_B_ZH)
            + len(FILLER_B_EN.findall(canon_spaced(text))))


def count_stutter(text: str) -> int:
    """連續重複的「處」數,不是字數。對對對對 = 1 處。"""
    zh = canon(text)
    return (len(STUTTER_ZH_2.findall(zh))
            + len(STUTTER_ZH_1.findall(zh))
            + len(STUTTER_EN.findall(canon_spaced(text))))


def drop_filler_a(canon_text: str) -> str:
    t = canon_text
    for f in FILLER_A_ZH:
        t = t.replace(f, "")
    return FILLER_A_EN.sub("", t)


def drop_filler_b(canon_text: str) -> str:
    t = canon_text
    for f in FILLER_B_ZH:
        t = t.replace(f, "")
    return FILLER_B_EN.sub("", t)


def collapse_stutter(canon_text: str) -> str:
    """對對對對 → 對,那個那個 → 那個,減減少 → 減少。

    注意這會誤傷合法的疊字(看看、謝謝、爸爸)。用在 halluc 的「來源變體」是
    安全的(變體只會讓判定更寬鬆),用在計分要小心。"""
    t = STUTTER_ZH_2.sub(r"\1", canon_text)
    t = STUTTER_ZH_1.sub(r"\1", t)
    return STUTTER_EN.sub(r"\1", t)
