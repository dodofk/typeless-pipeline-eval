#!/usr/bin/env python3
"""gt.py <wav> [wav ...] — 一個音檔丟給多個 ASR,標出「大家都同意」和「有分歧」的地方。

同意的地方 ≈ 就是 ground truth,不用人看。
分歧的地方 ← 人只需要看這裡。

用法:
    python3 gt.py audio/foo.wav                 # 三個引擎
    python3 gt.py --engines whisper,funasr x.wav  # 只跑本機
    python3 gt.py --gold gold/long2_technical.txt tts/mm28_long2_technical_f10.wav
                                                # 有標準答案時,順便算 ε
"""
import sys, os, re, json, time, difflib, pathlib, mimetypes, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

try:
    import zhconv
except ImportError:
    sys.exit("pip3 install zhconv")

# ---------------------------------------------------------------- 引擎定義
# 每家的 multipart 欄位名、認證方式、回傳 JSON 的 key 都不一樣，所以每個引擎
# 自己描述自己，不要硬套 OpenAI 的形狀。
ENGINES = {
    # 本機 whisper-large-v3-turbo（whisper.cpp server）
    "whisper": dict(
        url="http://127.0.0.1:8178/inference",
        fields={}, text_key="text"),

    # 本機 Fun-ASR：SenseVoice encoder + Qwen3-0.6B decoder
    # prompt 一定要跟 app 送的一致，不然結果會差 8–28% —— 純測量誤差。
    "funasr": dict(
        url="http://127.0.0.1:8379/v1/audio/transcriptions",
        fields={"model": "funasr-q4km", "response_format": "json",
                "prompt": os.environ.get("FUNASR_APP_PROMPT", "OpenWhispr")},
        text_key="text"),

    # 雲端 gpt-4o-transcribe：多模態 LLM，跟上面兩個誤差不相關
    "gpt4o": dict(
        url="https://api.openai.com/v1/audio/transcriptions",
        fields={"model": "gpt-4o-transcribe", "response_format": "json", "temperature": "0"},
        auth=("Authorization", "Bearer {}"), env="OPENAI_API_KEY", text_key="text"),

    # 雲端 ElevenLabs Scribe v1：又一個獨立家族，中英夾雜表現好
    # 注意欄位叫 model_id 不是 model，認證是 xi-api-key 不是 Bearer。
    "scribe": dict(
        url="https://api.elevenlabs.io/v1/speech-to-text",
        fields={"model_id": "scribe_v1", "tag_audio_events": "false"},
        auth=("xi-api-key", "{}"), env="ELEVENLABS_API_KEY", text_key="text"),
}

def post_audio(cfg, wav, timeout=300):
    """最小 multipart，不依賴 requests。"""
    boundary = "----gtboundary7f3a"
    blob = wav.read_bytes()
    ctype = mimetypes.guess_type(str(wav))[0] or "audio/wav"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode() + blob + b"\r\n"
    ]
    for k, v in cfg["fields"].items():
        if v == "":
            continue
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(cfg["url"], data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if cfg.get("auth"):
        header, tmpl = cfg["auth"]
        req.add_header(header, tmpl.format(os.environ[cfg["env"]]))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        return json.loads(raw).get(cfg["text_key"], "").strip()
    except json.JSONDecodeError:
        return raw.strip()

def run(name, wav):
    cfg = ENGINES[name]
    if cfg.get("env") and not os.environ.get(cfg["env"]):
        return name, None, 0.0, f"{cfg['env']} 沒設"
    t0 = time.monotonic()
    try:
        return name, post_audio(cfg, wav), time.monotonic() - t0, None
    except Exception as e:
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            detail = ": " + e.read().decode("utf-8", "replace")[:250]
        return name, None, time.monotonic() - t0, f"{type(e).__name__}{detail}"

# ---------------------------------------------------------------- 正規化 & 切詞
PUNCT = re.compile(r"[\s，。、？！；：「」『』（）〈〉《》,.\?!;:\"'()\-—…·]+")
TOKEN = re.compile(r"[一-鿿]|[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|%")

# 中文數字 → 阿拉伯數字。不做這步的話,whisper/gpt4o 輸出「0.3」、funasr 輸出
# 「零點三」、gold 寫「零點三」,CER 會把純粹的格式差異算成辨識錯誤。
_D = {"零":0,"一":1,"二":2,"兩":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
_U = {"十":10,"百":100,"千":1000}
_NUMRUN = re.compile(r"[零一二兩三四五六七八九十百千點]+")

def _parse(run):
    if any(u in run for u in _U):          # 位置制:二十五 → 25、六十 → 60
        total, cur = 0, 0
        for ch in run:
            if ch in _D: cur = _D[ch]
            elif ch in _U:
                total += (cur or 1) * _U[ch]; cur = 0
            else: return None
        return str(total + cur)
    return "".join(str(_D[c]) for c in run)  # 讀數制:一零二四 → 1024

def _zhnum(s):
    def rep(m):
        run = m.group(0)
        if run in ("一", "兩") or run == "點": return run   # 「說明一下」不要動
        parts = run.split("點")
        try:
            vals = [_parse(p) for p in parts if p]
            if any(v is None for v in vals): return run
        except KeyError:
            return run
        if len(parts) == 2 and parts[0] and parts[1]:
            return vals[0] + "." + vals[1]
        return vals[0] if vals else run
    return _NUMRUN.sub(rep, s)

def canon(s):
    """繁簡統一、英文小寫、去標點空白、中文數字轉阿拉伯數字。

    小數點要保護:PUNCT 會吃掉 "."，讓 whisper 的 "0.3" 變成 "03"，
    而 funasr 的「零點三」轉出來是 "0.3"，兩邊就對不起來了。"""
    t = zhconv.convert(s.strip(), "zh-tw").lower()
    t = re.sub(r"(?<=\d)\.(?=\d)", "\x00", t)
    return _zhnum(PUNCT.sub("", t)).replace("\x00", ".")

def tokens(s):
    """給對齊和顯示用:中文一字一 token,英數整串一 token。"""
    return TOKEN.findall(canon(s))

def show(tks):
    out = ""
    for t in tks:
        if out and re.match(r"[a-z]", t) and re.match(r"[a-z]", out[-1]):
            out += " "
        out += t
    return out

def cer(a, b):
    """字元級 —— 這樣 top-K / TopK / top k、BGE-M3 / BGE M3 不會被算成錯。
    詞級只拿來做對齊顯示,不拿來算分數。"""
    r, h = canon(a).replace(" ", ""), canon(b).replace(" ", "")
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(r), 1)

# ---------------------------------------------------------------- 對齊
def posmap(a, b):
    """a 的每個位置 → b 的對應位置。"""
    m = [0] * (len(a) + 1)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
        else:
            for i in range(i1, i2):
                m[i] = j1
    m[len(a)] = len(b)
    return m

def diff_spans(a, b):
    """回傳 a 座標上「跟 b 不一樣」的區間。"""
    return [(i1, i2) for tag, i1, i2, _, _ in
            difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal"]

def consensus(named):
    """named = [(engine, tokens), ...],第一個當骨架。

    回傳 (merged_tokens, rows),rows 是每個分歧點的 (a_span, {engine: text}, kind)。
    kind: 'majority' = 有兩個引擎一致(自動採用);'split' = 三個都不同(要人看)。
    """
    spine_name, spine = named[0]
    others = named[1:]
    maps = {n: posmap(spine, t) for n, t in others}

    marks = [False] * len(spine)
    for n, t in others:
        for i1, i2 in diff_spans(spine, t):
            for i in range(i1, min(i2, len(spine))):
                marks[i] = True
            if i1 == i2 and i1 < len(spine):   # 純插入:標記接縫
                marks[i1] = True

    # 把相鄰的分歧點合併成區間
    spans, i = [], 0
    while i < len(spine):
        if marks[i]:
            j = i
            while j < len(spine) and marks[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1

    merged, rows, cur = [], [], 0
    for i1, i2 in spans:
        merged.extend(spine[cur:i1])
        cands = {spine_name: spine[i1:i2]}
        for n, t in others:
            cands[n] = t[maps[n][i1]:maps[n][i2]]
        keys = [show(v) for v in cands.values()]
        tally = sorted({k: keys.count(k) for k in keys}.items(),
                       key=lambda kv: -kv[1])
        vote = tally[0][0]
        # 平手不能算多數決。2-2 的時候 max() 會照字典順序默默選一個，
        # 看起來像「自動決定」，其實是擲骰子 —— 實測 id14 就這樣把
        # 正確的「簡報」丟掉、選了「節目」。平手一律丟給人看。
        top = tally[0][1]
        runner_up = tally[1][1] if len(tally) > 1 else 0
        kind = "majority" if top >= 2 and top > runner_up else "split"
        merged.extend(tokens(vote) if kind == "majority" else ["⁇"])
        rows.append(((i1, i2), cands, kind, vote))
        cur = i2
    merged.extend(spine[cur:])
    return merged, rows

# ---------------------------------------------------------------- main
def main(argv):
    engines = list(ENGINES)
    gold = None
    wavs = []
    it = iter(range(len(argv)))
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--engines":
            i += 1; engines = argv[i].split(",")
        elif a == "--gold":
            i += 1; gold = pathlib.Path(argv[i])
        else:
            wavs.append(pathlib.Path(a))
        i += 1
    if not wavs:
        sys.exit(__doc__)

    for wav in wavs:
        if not wav.exists():
            print(f"!! 找不到 {wav}"); continue
        print(f"\n{'='*78}\n  {wav}\n{'='*78}")

        with ThreadPoolExecutor(max_workers=len(engines)) as ex:
            results = list(ex.map(lambda n: run(n, wav), engines))

        ok = []
        for name, text, dt, err in results:
            if err:
                print(f"\n[{name}] ✗ {dt:.1f}s  {err}")
            else:
                print(f"\n[{name}] {dt:.1f}s\n  {text}")
                ok.append((name, text))

        if len(ok) < 2:
            print("\n少於兩個引擎成功,無法比對。"); continue

        # 兩兩之間差多少 —— 全部很小 = 這段音檔簡單,共識可信
        print(f"\n--- 引擎兩兩差異 (token error rate) ---")
        for x in range(len(ok)):
            for y in range(x + 1, len(ok)):
                print(f"  {ok[x][0]:>8} vs {ok[y][0]:<8} {cer(ok[x][1], ok[y][1])*100:5.1f}%")

        named = [(n, tokens(t)) for n, t in ok]
        merged, rows = consensus(named)
        splits = [r for r in rows if r[2] == "split"]

        note = ("(只有兩個引擎 → 不可能有多數決,每一處都要人看)" if len(ok) < 3
                else f"({len(splits)} 處全部各說各話,要人看)")
        print(f"\n--- 分歧點 {len(rows)} 處 {note} ---")
        for (s, cands, kind, vote) in rows:
            flag = "⁇" if kind == "split" else " ·"
            print(f" {flag} @{s[0]:<4} " + " | ".join(
                f"{n}={show(v) or '∅'}" for n, v in cands.items())
                + ("" if kind == "split" else f"   → {vote}"))

        cov = 1 - sum(e - s for (s, e), *_ in rows) / max(len(named[0][1]), 1)
        print(f"\n自動決定的比例: {cov*100:.1f}%   人要看的: {len(splits)} 處")
        print(f"\n--- 合併結果 (⁇ = 要人填) ---\n{show(merged)}")

        if gold and gold.exists():
            g = gold.read_text()
            print(f"\n--- 跟標準答案比 (ε,越小代表共識越可信) ---")
            print(f"  {'共識':>8}          {cer(g, show(merged))*100:5.1f}%   ← ε")
            for n, t in ok:
                print(f"  {n:>8}          {cer(g, t)*100:5.1f}%")

if __name__ == "__main__":
    main(sys.argv[1:])
