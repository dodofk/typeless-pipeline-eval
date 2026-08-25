"""正規化 —— 全 codebase 唯一的一份。

坑 #1:以前 `score.py:norm()` 和 `gt.py:canon()` 各寫一份而且不一致 ——
score.py 沒有中文數字轉換,同一段文字兩支算出來的 CER 不一樣。
這裡以 gt.py 的 canon() 為準(那份是對的),score.py 那份廢掉。

坑 #2:CER 必須對繁簡中性 —— ref 和 hyp 兩邊都要轉 zh-tw。字體差異是下游 LLM
免費修得掉的,內容錯誤修不掉,所以選型要看內容不要看字體。

坑 #4:數 filler 之前一定要先去標點。曾經誤報「移除 64% 語助詞」,實際上模型把
`對對對對` 變成 `對,對,對,對` —— 逗號插進去,重複一個都沒少。
去標點這件事由 strip_punct() 提供,filler / stutter 的計數一律先過它。
"""

import re
import unicodedata

import zhconv

# ---------------------------------------------------------------- 基本樣式
# 注意:'%' 不在 PUNCT 裡(舊 score.py 把它吃掉了,導致「78.5%」跟「78.5」無法區分)。
PUNCT = re.compile(r"[\s，。、？！；：「」『』（）〈〉《》,.\?!;:\"'()\-—…·]+")

# 中文一字一 token;英數整串一 token;小數當一個 token;% 自成一個 token。
TOKEN = re.compile(r"[一-鿿]|[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|%")

CJK = re.compile(r"[一-鿿]")

_DEC = "\x00"  # 小數點的暫存哨兵,PUNCT 掃過之後再換回來

# ---------------------------------------------------------------- 中文數字
# 不做這步的話,whisper/gpt4o 輸出「0.3」、funasr 輸出「零點三」、gold 寫「零點三」,
# CER 會把純粹的格式差異算成辨識錯誤。
_D = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
      "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_U = {"十": 10, "百": 100, "千": 1000}
_NUMRUN = re.compile(r"[零一二兩三四五六七八九十百千點]+")


def _parse(run: str) -> str | None:
    """一段純中文數字 → 阿拉伯數字字串;認不出來回 None。"""
    if any(u in run for u in _U):          # 位置制:二十五 → 25、六十 → 60
        total, cur = 0, 0
        for ch in run:
            if ch in _D:
                cur = _D[ch]
            elif ch in _U:
                total += (cur or 1) * _U[ch]
                cur = 0
            else:
                return None
        return str(total + cur)
    if not all(c in _D for c in run):      # 讀數制:一零二四 → 1024
        return None
    return "".join(str(_D[c]) for c in run)


def zhnum(s: str) -> str:
    """中文數字 → 阿拉伯數字。認不出來的原樣留著,不要猜。

    「點」只有在兩側都是數字時才是小數點:「零點三」→ 0.3。
    「兩點半」「十點」的「點」是時間,不是小數點 —— 舊 gt.py 會把「兩點」整段
    吞成 "2"(半 留下來變「2半」),然後跟 score.py 期待的「2點半」對不起來。
    """
    def rep(m):
        run = m.group(0)
        if run in ("一", "兩", "點"):      # 「說明一下」不要動
            return run
        parts = run.split("點")
        if len(parts) == 2 and parts[0] and parts[1]:
            a, b = _parse(parts[0]), _parse(parts[1])
            if a is not None and b is not None:
                return a + "." + b          # 真的小數
        # 其餘:每段各自轉,「點」原樣留在原位
        out = []
        for part in parts:
            v = _parse(part) if part else None
            out.append(v if v is not None else part)
        return "點".join(out)
    return _NUMRUN.sub(rep, s)


# ---------------------------------------------------------------- 正規化層
def fold_zh(s: str) -> str:
    """繁簡統一到 zh-tw。CER 的前提(坑 #2)。"""
    return zhconv.convert(s, "zh-tw")


def strip_punct(s: str) -> str:
    """去標點與空白。小數點要保護 —— PUNCT 會吃掉 '.',讓 whisper 的 '0.3' 變
    '03',而「零點三」轉出來是 '0.3',兩邊就對不起來了。"""
    s = re.sub(r"(?<=\d)\.(?=\d)", _DEC, s)
    return PUNCT.sub("", s).replace(_DEC, ".")


def canon(s: str) -> str:
    """全套:Unicode 正規化 → 繁簡統一 → 小寫 → 去標點 → 中文數字轉阿拉伯。

    這就是舊的 gt.py:canon()(加上 NFKC 與 % 的處理)。CER、術語比對、
    幻覺偵測全部走這一條,不准另外再寫一份。"""
    t = unicodedata.normalize("NFKC", s.strip())
    t = fold_zh(t).lower()
    return zhnum(strip_punct(t))


def canon_spaced(s: str) -> str:
    """跟 canon() 一樣,但標點換成單一空格而不是刪掉 —— 保住英文的詞界。

    為什麼要兩種:canon() 去空白是為了讓「top k / top-K / topk」算同一個東西(CER 用),
    但那會把 "um, I think" 壓成 "umithink",英文的 \b 詞界全毀,所有英文 regex
    metric 就永遠算出 0。中文的指標要無空白視角(坑 #4 數重複字),英文的要有詞界視角。
    """
    t = unicodedata.normalize("NFKC", s.strip())
    t = fold_zh(t).lower()
    t = re.sub(r"(?<=\d)\.(?=\d)", _DEC, t)
    t = PUNCT.sub(" ", t).replace(_DEC, ".")
    return zhnum(re.sub(r"\s+", " ", t).strip())


def tokens(s: str) -> list[str]:
    """給對齊和顯示用:中文一字一 token,英數整串一 token。"""
    return TOKEN.findall(canon(s))


def show(tks: list[str]) -> str:
    """token list → 可讀字串(英文之間補空格)。"""
    out = ""
    for t in tks:
        if out and re.match(r"[a-z]", t) and re.match(r"[a-z]", out[-1]):
            out += " "
        out += t
    return out


# ---------------------------------------------------------------- 距離
def edit_distance(r: str, h: str) -> int:
    """Levenshtein。兩條 row 滾動,O(min) 記憶體。"""
    if len(r) < len(h):
        r, h = h, r
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """字元級錯誤率。字元級是刻意的 —— 這樣 top-K / TopK / top k、
    BGE-M3 / BGE M3 不會被算成錯。詞級只拿來做對齊顯示,不拿來算分數。"""
    r, h = canon(ref).replace(" ", ""), canon(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


# ---------------------------------------------------------------- 字體判定
def simplified_chars(s: str) -> list[str]:
    """回傳「簡體才有」的字。zh-tw 轉換後會變的那些就是。"""
    return [c for c in s if CJK.match(c) and c != zhconv.convert(c, "zh-tw")]


def script_of(s: str, threshold: int = 3) -> str:
    """整段是繁還是簡。門檻沿用舊 score.py:zh() 的 3 個字。"""
    return "簡" if len(simplified_chars(s)) > threshold else "繁"
