#!/usr/bin/env python3
"""grade.py —— 獨立評分腳本:吃檔案,吐分數。不連任何東西。

跟 ./tl 的差別:./tl 會自己去跑模型、去 OpenWhispr 的 SQLite 撈結果。
這支不會。它只讀你給的路徑,所以任何 pipeline 只要把結果轉成下面的格式就能評。

    uv run python grade.py --asr  asr/whisper.jsonl \
                           --polish polish/qwen35-v2.jsonl \
                           --ref  evalset/

格式(JSONL,一行一個 item,用 id 對接):

  --asr FILE       ASR 逐字稿(潤稿層的 input 也是它)
                   {"id":"teach-01","text":"...","dur_s":106.8,"elapsed_s":12.4}
                   必填 id / text。dur_s、elapsed_s 有才算得出 RTF。

  --polish FILE    潤稿輸出
                   {"id":"teach-01","text":"...","input":"...",
                    "latency_s":15.0,"gen_tok":376,"tok_s":27.1,"prompt_tok":890}
                   必填 id / text。input 省略的話用 --asr 同 id 的 text。
                   →→ 潤稿層一半的指標(語助詞移除率、口吃、幻覺)是拿 output
                      跟 input 比出來的。沒有 input 這些全部變 None。

  --ref FILE|DIR   正解。JSONL:
                   {"id":"teach-01","asr_ref":"...","polish_ref":"...",
                    "terms":[["BM25","B M 25"],["int8"]],"dur_s":106.8}
                   全部欄位都是選填 —— 缺的指標回 None,不是 0。
                   也可以直接給 evalset/ 目錄(有 manifest.jsonl 就自動讀)。

三個路徑都可以改成「一個資料夾裝 <id>.txt」,懶得寫 JSONL 時用。

輸出:stdout 一張表;--out x.json 存完整的每題明細。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from typeless.dataset import Item                      # noqa: E402
from typeless.metrics import asr as m_asr              # noqa: E402
from typeless.metrics import polish as m_polish        # noqa: E402
from typeless.scoring import aggregate                 # noqa: E402


# ---------------------------------------------------------------- 讀檔

def _read_jsonl(p: pathlib.Path) -> list[dict]:
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p}:{n} 不是合法 JSON —— {e}")
        if not isinstance(o, dict):
            raise SystemExit(f"{p}:{n} 應該是一個 object,拿到 {type(o).__name__}")
        out.append(o)
    return out


def _read_dir(d: pathlib.Path, suffix: str = ".txt") -> list[dict]:
    """一個資料夾裝 <id>.txt。err_* 開頭的跳過(舊腳本會把錯誤訊息寫成檔案)。"""
    recs = []
    for p in sorted(d.glob(f"*{suffix}")):
        if p.name.startswith("err_"):
            continue
        recs.append({"id": p.name[: -len(suffix)], "text": p.read_text(encoding="utf-8").strip()})
    return recs


def load_texts(spec: str, what: str) -> dict[str, dict]:
    """回傳 {id: record}。record 至少有 text。"""
    p = pathlib.Path(spec)
    if not p.exists():
        raise SystemExit(f"--{what} 指到的路徑不存在:{p}")
    recs = _read_dir(p) if p.is_dir() else _read_jsonl(p)

    out: dict[str, dict] = {}
    for r in recs:
        i = r.get("id")
        if not i:
            raise SystemExit(f"--{what} 有一筆沒有 id:{json.dumps(r, ensure_ascii=False)[:80]}")
        if i in out:
            raise SystemExit(f"--{what} 的 id 重複:{i}")
        if "text" not in r:
            raise SystemExit(f"--{what} 的 {i} 沒有 text 欄位")
        out[i] = r
    if not out:
        raise SystemExit(f"--{what} 讀到 0 筆:{p}")
    return out


def load_ref(spec: str) -> dict[str, dict]:
    """正解。JSONL / 資料夾 / evalset 目錄(有 manifest.jsonl)三種都吃。"""
    p = pathlib.Path(spec)
    if not p.exists():
        raise SystemExit(f"--ref 指到的路徑不存在:{p}")

    if p.is_dir() and (p / "manifest.jsonl").exists():
        return _ref_from_manifest(p)

    recs = _read_dir(p, ".txt") if p.is_dir() else _read_jsonl(p)
    out = {}
    for r in recs:
        i = r.get("id")
        if not i:
            raise SystemExit("--ref 有一筆沒有 id")
        # 資料夾模式:一律當潤稿正解,因為那是最常見的用法
        if "text" in r and "polish_ref" not in r and "asr_ref" not in r:
            r = {"id": i, "polish_ref": r["text"]}
        out[i] = r
    return out


def _ref_from_manifest(root: pathlib.Path) -> dict[str, dict]:
    """evalset 的 manifest.jsonl。

    ⚠️ text/<id>.zh-tw.txt 是**逐字稿**(語助詞還在),掛在 asr_ref。
       拿它當潤稿正解會讓指標整個反過來 —— 什麼都不刪的模型 CER 最低。
       潤稿正解要等 manifest 的 gold_final 有值。
    """
    out = {}
    for m in _read_jsonl(root / "manifest.jsonl"):
        i = m["id"]
        asr_ref = None
        tp = m.get("transcript_zh_tw")
        if tp and not m.get("transcript_is_parent_full"):
            f = root / tp
            if f.exists():
                asr_ref = f.read_text(encoding="utf-8").strip()
        gf = m.get("gold_final")
        polish_ref = None
        if gf:
            f = root / gf
            polish_ref = f.read_text(encoding="utf-8").strip() if f.exists() else gf
        out[i] = {"id": i, "asr_ref": asr_ref, "polish_ref": polish_ref,
                  "dur_s": m.get("duration_sec"), "terms": m.get("terms") or [],
                  "tags": m.get("tags") or []}
    return out


def _terms(v) -> list[tuple[str, ...]]:
    return [tuple(t) if isinstance(t, (list, tuple)) else (t,) for t in (v or [])]


# ---------------------------------------------------------------- 評分

def grade(asr: dict, polish: dict, ref: dict, n_cjk=None, n_latin=None,
          use_judge: bool = False) -> dict:
    """回傳 {"asr": {...}, "polish": {...}, "coverage": {...}}。少的欄位就是 None。"""
    res: dict = {"coverage": {}}

    if asr:
        items = []
        for i, r in asr.items():
            rf = ref.get(i, {})
            it = Item(id=i, asr_ref=rf.get("asr_ref"),
                      dur_s=r.get("dur_s") or rf.get("dur_s"),
                      terms=_terms(rf.get("terms")), tags=rf.get("tags") or [])
            items.append({"id": i, "out": r["text"],
                          "metrics": m_asr.score(it, r["text"], r.get("elapsed_s"))})
        res["asr"] = {"items": items,
                      "aggregate": aggregate(SimpleNamespace(arm="asr", items=items))}
        res["coverage"]["asr_ref"] = sum(1 for i in asr if ref.get(i, {}).get("asr_ref"))

    if polish:
        items = []
        for i, r in polish.items():
            rf = ref.get(i, {})
            src = r.get("input") or (asr.get(i, {}) or {}).get("text", "")
            it = Item(id=i, raw=src, ref=rf.get("polish_ref"),
                      dur_s=r.get("dur_s") or rf.get("dur_s"),
                      terms=_terms(rf.get("terms")), tags=rf.get("tags") or [])
            tm = {k: r[k] for k in ("wall_s", "latency_s", "gen_tok", "tok_s", "prompt_tok")
                  if k in r}
            if "latency_s" in tm:
                tm["wall_s"] = tm.pop("latency_s")
            m = m_polish.score(it, r["text"], tm, n_cjk, n_latin)
            if "judge" in r:                       # 檔案裡自己帶結果就用它,不重打 API
                m["judge"] = r["judge"]
            elif use_judge and src:
                from typeless.judge import judge as _judge     # 只有要用才 import
                print(f"  judge {i} …", file=sys.stderr)
                m["judge"] = _judge(src, r["text"]).as_dict()
            items.append({"id": i, "raw": src, "out": r["text"], "metrics": m})
        res["polish"] = {"items": items,
                         "aggregate": aggregate(SimpleNamespace(arm="polish", items=items))}
        res["coverage"]["polish_input"] = sum(1 for it in items if it["raw"])
        res["coverage"]["polish_ref"] = sum(1 for i in polish if ref.get(i, {}).get("polish_ref"))

    return res


# ---------------------------------------------------------------- 印表

def _f(v, pct=False, nd=1):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.{nd}f}%"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def print_report(res: dict, per_item: bool) -> None:
    a = res.get("asr", {}).get("aggregate")
    if a:
        print("\n=== ASR ===")
        print(f"  n            {a['n']}")
        print(f"  CER          {_f(a.get('cer'), pct=True, nd=2)}")
        print(f"  術語召回      {_f(a.get('term_recall'), pct=True)}  {a.get('term_hit') or ''}")
        print(f"  長度比 vs 正解 {_f(a.get('len_ratio'), nd=3)}   (<0.9 = 有整句被吞)")
        print(f"  RTF          {_f(a.get('rtf'), nd=3)}")
        print(f"  簡體殘留      {a.get('simp_chars')} 字")

    p = res.get("polish", {}).get("aggregate")
    if p:
        print("\n=== POLISH ===")
        print(f"  n            {p['n']}")
        print(f"  語助詞 A 殘留  {p.get('filler_a_out')} / 輸入 {p.get('filler_a_in')}"
              f"   移除率 {_f(p.get('filler_a_removed'), pct=True)}")
        print(f"  語助詞 B 殘留  {p.get('filler_b_out')}")
        print(f"  口吃殘留      {p.get('stutter_out')}   移除率 {_f(p.get('stutter_removed'), pct=True)}")
        print(f"  簡體殘留      {p.get('simp_out')} 字")
        print(f"  長度比 vs 輸入 {_f(p.get('len_ratio_raw'), nd=3)}   (<0.7 = 刪過頭)")
        print(f"  幻覺率        {_f(p.get('halluc_rate'), pct=True, nd=2)}"
              f"   ({p.get('halluc_chars')} 字憑空出現)")
        print(f"  漂移(judge)  {_f(p.get('drift_high'))} high / {_f(p.get('drift_n'))} 全部"
              f"   已判 {_f(p.get('judged'))} 題"
              + (f"   ⚠️ {p['judge_err']} 題判定失敗(不列入)" if p.get("judge_err") else ""))
        print(f"  CER vs 正解   {_f(p.get('cer_ref'), pct=True, nd=2)}")
        print(f"  術語保留      {_f(p.get('term_keep'), pct=True)}")
        print(f"  速度          {_f(p.get('tok_s'), nd=2)} tok/s   延遲 {_f(p.get('latency_s'), nd=1)}s")

    c = res.get("coverage", {})
    if c:
        print("\n--- 覆蓋率(缺的指標會是 —,不是 0)---")
        if "polish_input" in c:
            n = res["polish"]["aggregate"]["n"]
            print(f"  潤稿 input   {c['polish_input']}/{n}"
                  + ("   ⚠️ 沒有 input 的題目算不出移除率/幻覺"
                     if c["polish_input"] < n else ""))
        if "asr_ref" in c:
            print(f"  ASR 正解     {c['asr_ref']}/{res['asr']['aggregate']['n']}")
        if "polish_ref" in c:
            print(f"  潤稿正解     {c['polish_ref']}/{res['polish']['aggregate']['n']}")

    if per_item and res.get("polish"):
        print("\n--- 每題(潤稿)---")
        print(f"{'id':<14}{'語助A':>7}{'口吃':>6}{'簡':>5}{'長度比':>8}{'幻覺':>8}{'CER':>8}")
        for it in res["polish"]["items"]:
            m = it["metrics"]
            print(f"{it['id']:<14}{m['filler_a']['out']:>7}{m['stutter']['out']:>6}"
                  f"{m['simp']['out']:>5}{_f(m.get('len_ratio_raw'), nd=2):>8}"
                  f"{_f((m.get('halluc') or {}).get('rate'), pct=True, nd=2):>8}"
                  f"{_f(m.get('cer_ref'), pct=True, nd=2):>8}")

    if per_item and res.get("asr"):
        print("\n--- 每題(ASR)---")
        print(f"{'id':<14}{'CER':>9}{'長度比':>9}{'RTF':>8}{'簡':>5}")
        for it in res["asr"]["items"]:
            m = it["metrics"]
            print(f"{it['id']:<14}{_f(m.get('cer'), pct=True, nd=2):>9}"
                  f"{_f(m.get('len_ratio'), nd=3):>9}{_f(m.get('rtf'), nd=3):>8}"
                  f"{m.get('simp_chars', 0):>5}")


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(
        description="評分。吃檔案,不連任何東西。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--asr", help="ASR 逐字稿 JSONL / 資料夾")
    ap.add_argument("--polish", help="潤稿輸出 JSONL / 資料夾")
    ap.add_argument("--ref", help="正解 JSONL / 資料夾 / evalset 目錄")
    ap.add_argument("--out", help="把完整明細寫成 JSON")
    ap.add_argument("--per-item", action="store_true", help="連每一題都印")
    ap.add_argument("--json", action="store_true", help="只印 JSON,不印表")
    ap.add_argument("--n-cjk", type=int, default=None, help="幻覺:中文 n-gram(預設 3)")
    ap.add_argument("--n-latin", type=int, default=None, help="幻覺:英文 n-gram(預設 4)")
    ap.add_argument("--judge", action="store_true",
                    help="加算語意漂移(LLM-as-a-judge)。⚠️ 只有這個選項會連網,"
                         "需要 MINIMAX_API_KEY。不加就完全離線。")
    a = ap.parse_args()

    if not a.asr and not a.polish:
        ap.error("--asr 跟 --polish 至少要給一個")

    asr = load_texts(a.asr, "asr") if a.asr else {}
    polish = load_texts(a.polish, "polish") if a.polish else {}
    ref = load_ref(a.ref) if a.ref else {}

    # 對帳:兩邊 id 對不起來要講,不能靜靜少算幾題
    if asr and polish:
        only_p = sorted(set(polish) - set(asr))
        if only_p:
            print(f"⚠️  {len(only_p)} 題只有潤稿沒有 ASR,無法算移除率/幻覺:"
                  f"{', '.join(only_p[:5])}{' …' if len(only_p) > 5 else ''}", file=sys.stderr)
    if ref:
        unknown = sorted((set(asr) | set(polish)) - set(ref))
        if unknown:
            print(f"⚠️  {len(unknown)} 題在 --ref 裡找不到:"
                  f"{', '.join(unknown[:5])}{' …' if len(unknown) > 5 else ''}", file=sys.stderr)

    res = grade(asr, polish, ref, a.n_cjk, a.n_latin, a.judge)

    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_report(res, a.per_item)

    if a.out:
        pathlib.Path(a.out).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
