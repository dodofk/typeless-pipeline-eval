#!/usr/bin/env python3
"""grade.py —— 評分。吃檔案,吐分數。除了 --judge 之外不連任何東西。

    uv run python grade.py --asr asr/breeze.jsonl --ref evalset/
    uv run python grade.py --asr asr/breeze.jsonl --polish polish/breeze-qwen35-v2.jsonl \
                           --ref evalset/ --judge --per-item

指標只有這幾個,兩層都一樣:

    CER        字元錯誤率。要有 reference 才算得出來
    runtime_s  這一題花了幾秒
    RTF        runtime_s / 音檔秒數。越小越快,<1 才跟得上說話
    judge      幻覺 + 品質(1-5),LLM 判的,要 --judge 才會跑

RAM / CPU / 溫度不在這裡 —— 那是跑的時候要監控的東西,用 monitor.py。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from typeless.norm import canon, cer as _cer          # noqa: E402


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


def _read_dir(d: pathlib.Path) -> list[dict]:
    return [{"id": p.stem, "text": p.read_text(encoding="utf-8").strip()}
            for p in sorted(d.glob("*.txt")) if not p.name.startswith("err_")]


def load_texts(spec: str, what: str) -> dict[str, dict]:
    p = pathlib.Path(spec)
    if not p.exists():
        raise SystemExit(f"--{what} 指到的路徑不存在:{p}")
    out: dict[str, dict] = {}
    for r in (_read_dir(p) if p.is_dir() else _read_jsonl(p)):
        i = r.get("id")
        if not i:
            raise SystemExit(f"--{what} 有一筆沒有 id")
        if i in out:
            raise SystemExit(f"--{what} 的 id 重複:{i}")
        if "text" not in r:
            raise SystemExit(f"--{what} 的 {i} 沒有 text 欄位")
        out[i] = r
    if not out:
        raise SystemExit(f"--{what} 讀到 0 筆:{p}")
    return out


def load_ref(spec: str) -> dict[str, dict]:
    p = pathlib.Path(spec)
    if not p.exists():
        raise SystemExit(f"--ref 指到的路徑不存在:{p}")
    if p.is_dir() and (p / "manifest.jsonl").exists():
        return _ref_from_manifest(p)
    out = {}
    for r in (_read_dir(p) if p.is_dir() else _read_jsonl(p)):
        i = r.get("id")
        if not i:
            raise SystemExit("--ref 有一筆沒有 id")
        if "text" in r and "asr_ref" not in r and "polish_ref" not in r:
            r = {"id": i, "polish_ref": r["text"]}      # 資料夾模式一律當潤稿正解
        out[i] = r
    return out


def _ref_from_manifest(root: pathlib.Path) -> dict[str, dict]:
    """evalset/manifest.jsonl。

    ⚠️ text/<id>.zh-tw.txt 是**逐字稿**(語助詞還在),掛在 asr_ref。
       拿它當潤稿正解會讓指標反過來 —— 什麼都不刪的模型 CER 最低。
       潤稿正解要等 manifest 的 gold_final 有值。
    """
    out = {}
    for m in _read_jsonl(root / "manifest.jsonl"):
        asr_ref = None
        tp = m.get("transcript_zh_tw")
        if tp and not m.get("transcript_is_parent_full"):
            f = root / tp
            asr_ref = f.read_text(encoding="utf-8").strip() if f.exists() else None
        polish_ref = None
        if m.get("gold_final"):
            f = root / m["gold_final"]
            polish_ref = f.read_text(encoding="utf-8").strip() if f.exists() else m["gold_final"]
        out[m["id"]] = {"id": m["id"], "asr_ref": asr_ref, "polish_ref": polish_ref,
                        "dur_s": m.get("duration_sec")}
    return out


# ---------------------------------------------------------------- 指標
def score_one(text: str, ref: str | None, runtime_s: float | None,
              dur_s: float | None) -> dict:
    """一題的三個指標。算不出來的回 None,不回 0 —— 「沒量到」不是「量到 0」。"""
    return {
        "cer": round(_cer(ref, text), 4) if ref else None,
        "runtime_s": round(runtime_s, 3) if runtime_s else None,
        "rtf": round(runtime_s / dur_s, 4) if (runtime_s and dur_s) else None,
        "n_chars": len(canon(text)),
    }


def _mean(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    return round(st.mean(v), 4) if v else None


def aggregate(items: list[dict]) -> dict:
    """比率取平均、時間取總和。混著用會誤導。"""
    m = [i["metrics"] for i in items]
    rt = [x["runtime_s"] for x in m if x["runtime_s"]]
    jud = [x.get("judge") for x in m if x.get("judge") and not x["judge"].get("error")]
    jerr = sum(1 for x in m if x.get("judge") and x["judge"].get("error"))
    return {
        "n": len(items),
        "n_scored": sum(1 for x in m if x["cer"] is not None),
        "cer": _mean(x["cer"] for x in m),
        "cer_median": round(st.median([x["cer"] for x in m if x["cer"] is not None]), 4)
                      if any(x["cer"] is not None for x in m) else None,
        "rtf": _mean(x["rtf"] for x in m),
        "rtf_median": round(st.median([x["rtf"] for x in m if x["rtf"]]), 4)
                      if any(x["rtf"] for x in m) else None,
        "runtime_total_s": round(sum(rt), 1) if rt else None,
        "runtime_mean_s": _mean(rt),
        "runtime_max_s": round(max(rt), 1) if rt else None,
        # judge
        "quality": _mean(j.get("quality") for j in jud) if jud else None,
        "quality_min": min((j["quality"] for j in jud if j.get("quality")), default=None),
        "halluc_items": sum(1 for j in jud if j.get("n_halluc")) or None,
        "halluc_high": sum(j.get("n_high", 0) for j in jud) or None,
        "judged": len(jud) or None,
        "judge_err": jerr or None,
    }


# ---------------------------------------------------------------- 主流程
def grade(asr: dict, polish: dict, ref: dict, use_judge: bool, jcfg) -> dict:
    from typeless.judge import judge as run_judge

    res: dict = {}
    if asr:
        items = []
        for i, r in asr.items():
            rf = ref.get(i, {})
            items.append({"id": i, "metrics": score_one(
                r["text"], rf.get("asr_ref"), r.get("elapsed_s"),
                r.get("dur_s") or rf.get("dur_s"))})
        res["asr"] = {"items": items, "aggregate": aggregate(items)}

    if polish:
        items = []
        for i, r in polish.items():
            rf = ref.get(i, {})
            src = r.get("input") or (asr.get(i, {}) or {}).get("text", "")
            m = score_one(r["text"], rf.get("polish_ref"),
                          r.get("latency_s") or r.get("elapsed_s"),
                          r.get("dur_s") or rf.get("dur_s"))
            if "judge" in r:                       # 檔案裡帶了就用,不重打 API
                m["judge"] = r["judge"]
            elif use_judge:
                if not src:
                    m["judge"] = {"error": "沒有 input,judge 需要對照原文"}
                else:
                    print(f"  judge {i} …", file=sys.stderr)
                    m["judge"] = run_judge(src, r["text"], jcfg).as_dict()
            items.append({"id": i, "input_chars": len(src), "metrics": m})
        res["polish"] = {"items": items, "aggregate": aggregate(items)}

    return res


# ---------------------------------------------------------------- 印表
def _f(v, pct=False, nd=2, unit=""):
    if v is None:
        return "—"
    return f"{v * 100:.{nd}f}%" if pct else f"{v:.{nd}f}{unit}"


def _section(name: str, a: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  n              {a['n']}"
          + (f"   (有正解可算 CER 的:{a['n_scored']})" if a['n_scored'] != a['n'] else ""))
    print(f"  CER            平均 {_f(a['cer'], pct=True)}   中位 {_f(a['cer_median'], pct=True)}")
    print(f"  RTF            平均 {_f(a['rtf'], nd=3)}   中位 {_f(a['rtf_median'], nd=3)}"
          "        (處理秒數/音檔秒數)")
    print(f"  runtime        總計 {_f(a['runtime_total_s'], nd=1, unit='s')}"
          f"   平均 {_f(a['runtime_mean_s'], nd=1, unit='s')}"
          f"   最慢 {_f(a['runtime_max_s'], nd=1, unit='s')}")
    if a["judged"] or a["judge_err"]:
        print(f"  品質(1-5)     平均 {_f(a['quality'], nd=2)}   最低 {a['quality_min'] or '—'}"
              f"        已判 {a['judged'] or 0} 題"
              + (f"   ⚠️ {a['judge_err']} 題判定失敗(不列入)" if a["judge_err"] else ""))
        print(f"  幻覺           {a['halluc_items'] or 0} 題有"
              f"   其中 high {a['halluc_high'] or 0} 處")


def print_report(res: dict, per_item: bool) -> None:
    for key, name in (("asr", "ASR"), ("polish", "POLISH")):
        if key in res:
            _section(name, res[key]["aggregate"])

    if not per_item:
        return
    for key, name in (("asr", "ASR"), ("polish", "POLISH")):
        if key not in res:
            continue
        print(f"\n--- 每題({name})---")
        has_j = any(i["metrics"].get("judge") for i in res[key]["items"])
        head = f"{'id':<13}{'CER':>9}{'runtime':>10}{'RTF':>8}"
        if has_j:
            head += f"{'品質':>6}{'幻覺':>6}{'high':>6}"
        print(head)
        for it in res[key]["items"]:
            m = it["metrics"]
            row = (f"{it['id']:<13}{_f(m['cer'], pct=True):>9}"
                   f"{_f(m['runtime_s'], nd=1, unit='s'):>10}{_f(m['rtf'], nd=3):>8}")
            j = m.get("judge")
            if has_j:
                if j and not j.get("error"):
                    row += f"{j.get('quality') or '—':>6}{j.get('n_halluc', 0):>6}{j.get('n_high', 0):>6}"
                else:
                    row += f"{'—':>6}{'—':>6}{'—':>6}"
            print(row)


def print_halluc(res: dict) -> None:
    items = [i for i in res.get("polish", {}).get("items", [])
             if (i["metrics"].get("judge") or {}).get("hallucinations")]
    if not items:
        return
    print("\n--- judge 抓到的幻覺 ---")
    for it in items:
        j = it["metrics"]["judge"]
        print(f"\n[{it['id']}]  品質 {j.get('quality')} — {j.get('quality_why', '')}")
        for h in j["hallucinations"]:
            print(f"  ({h.get('severity')}/{h.get('kind')}) 「{str(h.get('span'))[:70]}」")
            if h.get("basis"):
                print(f"     來源應該是:「{str(h['basis'])[:70]}」")
            print(f"     {h.get('why', '')}")


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(
        description="評分:CER / runtime / RTF / judge(幻覺 + 品質)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--asr", help="ASR 逐字稿 JSONL 或資料夾")
    ap.add_argument("--polish", help="潤稿輸出 JSONL 或資料夾")
    ap.add_argument("--ref", help="正解 JSONL / 資料夾 / evalset 目錄")
    ap.add_argument("--out", help="完整明細寫成 JSON")
    ap.add_argument("--per-item", action="store_true")
    ap.add_argument("--json", action="store_true", help="只印 JSON")
    ap.add_argument("--judge", action="store_true",
                    help="跑 LLM judge(幻覺 + 品質)。⚠️ 只有這個選項會連網")
    ap.add_argument("--judge-model", default="", help="預設讀 JUDGE_MODEL,再退回 gpt-5.4-mini")
    ap.add_argument("--judge-base-url", default="", help="預設讀 JUDGE_BASE_URL,再退回 OpenAI")
    ap.add_argument("--judge-url", default="", help="完整 endpoint 覆寫(路徑不標準的供應商)")
    a = ap.parse_args()

    if not a.asr and not a.polish:
        ap.error("--asr 跟 --polish 至少要給一個")

    asr = load_texts(a.asr, "asr") if a.asr else {}
    polish = load_texts(a.polish, "polish") if a.polish else {}
    ref = load_ref(a.ref) if a.ref else {}

    if asr and polish:
        only_p = sorted(set(polish) - set(asr))
        if only_p:
            print(f"⚠️  {len(only_p)} 題只有潤稿沒有 ASR:{', '.join(only_p[:5])}", file=sys.stderr)
    if ref:
        unknown = sorted((set(asr) | set(polish)) - set(ref))
        if unknown:
            print(f"⚠️  {len(unknown)} 題在 --ref 裡找不到:{', '.join(unknown[:5])}", file=sys.stderr)

    jcfg = None
    if a.judge:
        from typeless.judge import JudgeConfig
        jcfg = JudgeConfig(model=a.judge_model, base_url=a.judge_base_url, url=a.judge_url)
        url, model, key = jcfg.resolved()
        print(f"judge:{model} @ {url}" + ("" if key else "   ⚠️ 沒有 API key"), file=sys.stderr)

    res = grade(asr, polish, ref, a.judge, jcfg)

    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_report(res, a.per_item)
        print_halluc(res)

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
