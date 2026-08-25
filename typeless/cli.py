"""tl —— local-typeless 的 metric CLI。

    ./tl run     跑潤稿 + 計分 + 存成一筆 run
    ./tl score   對既有 run 重算分數(不重跑模型)
    ./tl judge   對既有 run 補跑語意漂移判定
    ./tl report  跨 run 對照表
    ./tl show    看單一 run/clip 的輸入輸出與被標記的片段
    ./tl list    列出所有 run
    ./tl legacy  把 out/ 的歷史輸出匯成 run(重現舊腳本的表)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import asr_sources
from . import dataset as ds
from . import engines, judge as judge_mod, report as report_mod, scoring
from .metrics import halluc
from .registry import ROOT, Run, make_run_id, env_snapshot, read_index, rebuild_index


# ---------------------------------------------------------------- run
def cmd_run(a) -> int:
    data = ds.load(a.dataset)

    # --- ASR hypothesis 注入 ---------------------------------------------
    # raw 屬於 run 不屬於 dataset:同一批音檔換一個 ASR 模型就是另一份 hypothesis。
    asr_meta = None
    if a.asr_source:
        texts, asr_meta = asr_sources.resolve(a.asr_source, evalset_root=a.dataset)
        rec = data.attach_asr(texts)
        print(f"ASR 來源 {a.asr_source}:對上 {len(rec['matched'])} 個", file=sys.stderr)
        if rec["dataset_without_asr"]:
            print(f"  ⚠️  dataset 有但這個 ASR 來源沒有:{', '.join(rec['dataset_without_asr'])}",
                  file=sys.stderr)
        if rec["asr_without_dataset"]:
            print(f"  ⚠️  ASR 有但 dataset 沒有(不會計分):{', '.join(rec['asr_without_dataset'])}",
                  file=sys.stderr)
        if asr_meta.get("duplicates"):
            print(f"  ⚠️  同一個檔有多筆,取最新:{', '.join(asr_meta['duplicates'])}",
                  file=sys.stderr)
        if asr_meta.get("warning"):
            print(f"  ⚠️  {asr_meta['warning']}", file=sys.stderr)
    elif a.input != "gold" and any(i.needs_asr_source for i in data.items):
        sys.exit(
            "這個 dataset 沒有內建 ASR 逐字稿,要用 --asr-source 指定被測的 ASR 輸出。\n"
            "  --asr-source openwhispr:<folder>   OpenWhispr audio upload 的結果\n"
            "  --asr-source openwhispr-polished   OpenWhispr 自己的 AI 潤稿(對照組)\n"
            "  --asr-source spokenly              Spokenly/ElevenLabs 原文\n"
            "                                     ⚠️ 那是 ground truth 的來源引擎,\n"
            "                                        分數是上界不是實力\n"
            "  --asr-source dir:<path>            一個目錄的 <id>.txt")

    # gold-input arm(brief §3):餵人工逐字稿而不是 ASR 輸出,把兩層誤差切開。
    # 同一個潤稿模型跑兩次,差值就是「ASR 的錯誤讓潤稿層額外壞了多少」。
    if a.input == "gold":
        missing = [i.id for i in data.items if not i.asr_ref]
        if missing:
            sys.exit(f"--input gold 需要人工逐字稿,這些 item 沒有:{', '.join(missing)}\n"
                     f"  evalset 放在 text/<id>-asr.txt")
        for i in data.items:
            i.raw = i.asr_ref
    if a.only:
        keep = set(a.only.split(","))
        data.items = [i for i in data.items if i.id in keep]

    # 沒有 input、或 input 其實是母檔的完整逐字稿 → 跳過並說清楚為什麼。
    # 靜靜地少跑幾個 item 比報錯更糟 —— 表上的 n 會對不起來。
    skipped = [i for i in data.items if not i.usable]
    if skipped and not a.include_unusable:
        for i in skipped:
            why = "共用母檔的完整逐字稿,不是這一段的" if i.meta.get("parent_full") else "沒有 input 文字"
            print(f"跳過 {i.id}:{why}", file=sys.stderr)
        data.items = [i for i in data.items if i.usable]
    if not data.items:
        sys.exit("資料集是空的")

    cfg = engines.PolishConfig(
        url=a.url, model=a.model, prompt_file=a.prompt, temp=a.temp, seed=a.seed,
        top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_tokens,
        enable_thinking=a.thinking)
    prompt, cfg.prompt_sha = engines.load_prompt(cfg.prompt_file)

    env = env_snapshot()
    if env["llama_procs"] > 1:
        print(f"⚠️  偵測到 {env['llama_procs']} 個 llama-server 在跑。"
              f"記憶體頻寬會互搶,tok/s 不可跨 run 比(坑#7)。", file=sys.stderr)

    run = Run(run_id=make_run_id(a.label), label=a.label, arm="polish",
              dataset=data.name, env=env, notes=a.notes or "",
              config={"polish": cfg.as_dict(), "input": a.input, "asr": asr_meta})

    if not a.no_warm:
        print("暖機…", file=sys.stderr)
        engines.warm(cfg, prompt)

    for n, item in enumerate(data.items, 1):
        r = engines.polish(item.raw, cfg, prompt)
        if r.error:
            print(f"[{n}/{len(data.items)}] {item.id}  ✗ {r.error}", file=sys.stderr)
        else:
            t = r.timings
            print(f"[{n}/{len(data.items)}] {item.id}  {t.get('wall_s', 0):.1f}s  "
                  f"{t.get('gen_tok')} tok @ {t.get('tok_s') or 0:.1f} tok/s", file=sys.stderr)
        run.items.append({"id": item.id, "raw": item.raw, "out": r.text,
                          "timings": r.timings, "error": r.error})

    scoring.rescore(run, data)
    if a.judge:
        _judge_run(run, verbose=True)
        run.aggregate = scoring.aggregate(run)
    path = run.save()
    print(f"\n→ {path}", file=sys.stderr)
    print(report_mod.report(arm="polish", label_like=a.label))
    return 0


# ---------------------------------------------------------------- score
def cmd_score(a) -> int:
    run = Run.load(a.run_id)
    data = ds.load(a.dataset or run.dataset)
    if a.n_cjk or a.n_latin:
        halluc.N_CJK = a.n_cjk or halluc.N_CJK
        halluc.N_LATIN = a.n_latin or halluc.N_LATIN
        run.config.setdefault("metrics", {}).update(
            {"n_cjk": halluc.N_CJK, "n_latin": halluc.N_LATIN})
    before = dict(run.aggregate)
    scoring.rescore(run, data)
    run.save()
    print(f"重算完成:{run.run_id}")
    for k in sorted(set(before) | set(run.aggregate)):
        o, n = before.get(k), run.aggregate.get(k)
        if o != n:
            print(f"  {k:<20} {o} → {n}")
    return 0


# ---------------------------------------------------------------- judge
def _judge_run(run, verbose=False) -> None:
    cfg = judge_mod.JudgeConfig()
    run.config["judge"] = cfg.as_dict()
    for it in run.items:
        if not it.get("out"):
            continue
        r = judge_mod.judge(it["raw"], it["out"], cfg)
        it["judge"] = r.as_dict()
        it.setdefault("metrics", {})["judge"] = it["judge"]
        if verbose:
            mark = "✗" if r.error else f"{len(r.drifts)} drift / {r.high} high"
            print(f"  judge {it['id']}: {r.error or mark}", file=sys.stderr)


def cmd_judge(a) -> int:
    run = Run.load(a.run_id)
    _judge_run(run, verbose=True)
    run.aggregate = scoring.aggregate(run)
    run.save()
    for it in run.items:
        for d in (it.get("judge") or {}).get("drifts", []):
            print(f"[{it['id']}] {d.get('severity')}/{d.get('kind')}: "
                  f"{d.get('output_span', '')[:70]}\n     ← {d.get('input_basis', '')[:70]}"
                  f"\n     {d.get('why', '')}")
    return 0


# ---------------------------------------------------------------- report / list / show
def cmd_report(a) -> int:
    print(report_mod.report(arm=a.arm, dataset=a.dataset, since=a.since,
                            label_like=a.label))
    return 0


def cmd_list(a) -> int:
    rows = read_index()
    if not rows:
        print("還沒有任何 run。")
        return 0
    # 欄寬照實際內容算 —— 寫死 34 的話,長一點的 run_id 會跟 arm 黏在一起,
    # `./tl list | awk '{print $1}'` 就會撈到壞掉的 id。
    w = max((len(r["run_id"]) for r in rows), default=6) + 2
    print(f"{'run_id':<{w}}{'arm':<8}{'dataset':<16}{'n':>4}  label")
    for r in rows:
        print(f"{r['run_id']:<{w}}{r.get('arm', ''):<8}{r.get('dataset', ''):<16}"
              f"{r.get('n', 0):>4}  {r.get('label', '')}")
    return 0


def cmd_show(a) -> int:
    run = Run.load(a.run_id)
    items = [i for i in run.items if not a.item or i["id"] == a.item]
    if not items:
        sys.exit(f"run 裡沒有 item {a.item}")
    for it in items:
        m = it.get("metrics") or {}
        print(f"\n{'=' * 78}\n### {run.label} / {it['id']}")
        print(f"{'-' * 78}\nRAW    : {it['raw']}\n\nOUT    : {it.get('out', '')}")
        if it.get("error"):
            print(f"\n✗ {it['error']}")
            continue
        h = m.get("halluc") or {}
        print(f"\n[長度比 {m.get('len_ratio_raw')} | 語助A {(m.get('filler_a') or {}).get('in')}"
              f"→{(m.get('filler_a') or {}).get('out')} | 口吃 {(m.get('stutter') or {}).get('in')}"
              f"→{(m.get('stutter') or {}).get('out')} | 簡體 {(m.get('simp') or {}).get('in')}"
              f"→{(m.get('simp') or {}).get('out')}]")
        print(f"[幻覺 {(h.get('rate') or 0) * 100:.2f}%  被標記:{h.get('spans') or '無'}]")
        for d in (m.get("judge") or {}).get("drifts", []):
            print(f"[語意漂移 {d.get('severity')}/{d.get('kind')}] {d.get('output_span', '')}"
                  f"\n    ← {d.get('input_basis', '')}\n    {d.get('why', '')}")
    return 0


# ---------------------------------------------------------------- legacy 匯入
def cmd_legacy(a) -> int:
    """把 out/ 的歷史輸出匯成 run,證明新 pipeline 重現得出舊腳本的數字。

    完全不碰模型 —— 只是讀檔 + 用新 metric 重算。"""
    made = []

    if a.which in ("asr", "all"):
        # score.py 的表:out/<engine>_<tag>.txt vs gold/
        data = ds.load("legacy-tts")
        # err_*.txt 是 stderr log 不是逐字稿,混進來 CER 會變 913913%
        engines_seen = {p.name.rsplit("_mm28_", 1)[0].rsplit("_say_", 1)[0]
                        for p in (ROOT / "out").glob("*_long*.txt")
                        if not p.name.startswith("err_")}
        for eng in sorted(e for e in engines_seen if e and not e.startswith("err")):
            items = []
            for item in data.items:
                p = ROOT / "out" / f"{eng}_{item.id}.txt"
                if p.exists():
                    items.append({"id": item.id, "raw": "", "out": p.read_text().strip(),
                                  "timings": {}})
            if not items:
                continue
            run = Run(run_id=make_run_id(f"legacy-asr-{eng}", 0), label=f"legacy:{eng}",
                      arm="asr", dataset=data.name, items=items,
                      env={"note": "從 out/ 匯入的歷史輸出,沒有當時的環境資訊"},
                      config={"asr": {"engine": eng, "source": "out/*.txt"}},
                      notes="score.py 那張表的重現。ASR 由外部產生,這裡只重新計分。")
            scoring.rescore(run, data)
            run.save()
            made.append(run.run_id)

    if a.which in ("polish", "all"):
        # bench_polish.py 的表:out/e2e_<clip>_polished_<tag>.txt
        for name, glob, ds_name in (("zh", "e2e_*_polished_*.txt", "legacy-real"),
                                    ("en", "en_*_polished_*.txt", "legacy-real-en")):
            data = ds.load(ds_name)
            tags = set()
            for p in (ROOT / "out").glob(glob):
                stem = p.stem.split("_polished_", 1)
                if len(stem) == 2:
                    tags.add(stem[1])
            for tag in sorted(tags):
                items = []
                for item in data.items:
                    pre = "e2e" if name == "zh" else "en"
                    p = ROOT / "out" / f"{pre}_{item.id}_polished_{tag}.txt"
                    if p.exists():
                        items.append({"id": item.id, "raw": item.raw,
                                      "out": p.read_text().strip(), "timings": {}})
                if not items:
                    continue
                run = Run(run_id=make_run_id(f"legacy-{name}-{tag}", 0),
                          label=f"legacy:{name}/{tag}", arm="polish", dataset=ds_name,
                          items=items,
                          env={"note": "從 out/ 匯入,沒有當時的 timing 與獨佔資訊",
                               "speed_trustworthy": None},
                          config={"polish": {"model": tag, "prompt_file": None,
                                             "temp": None, "prompt_sha": None}},
                          notes="bench_polish.py 那張表的重現。只重新計分,不重跑模型。")
                scoring.rescore(run, data)
                run.save()
                made.append(run.run_id)

    print(f"匯入 {len(made)} 筆 run:")
    for r in made:
        print("  " + r)
    return 0


def cmd_reindex(a) -> int:
    print(f"重建 index:{rebuild_index()} 筆")
    return 0


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑潤稿 + 計分 + 存 run")
    r.add_argument("--dataset", default="evalset")
    r.add_argument("--label", required=True, help="這個 config 的名字,會進 run record")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--model", default="local")
    r.add_argument("--prompt", default="prompts/cleanup-zhTW-mixed-v2.txt")
    r.add_argument("--temp", type=float, default=0.0, help="A/B 一定要 0(坑#3)")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--max-tokens", type=int, default=2048)
    r.add_argument("--thinking", action="store_true", help="開 reasoning(潤稿不需要,慢 39 倍)")
    r.add_argument("--no-warm", action="store_true")
    r.add_argument("--judge", action="store_true", help="順便跑語意漂移判定(要 API key)")
    r.add_argument("--asr-source",
                   help="被測的 ASR 輸出從哪來:openwhispr[:folder] / "
                        "openwhispr-polished / spokenly[:stage] / dir:<path>")
    r.add_argument("--input", choices=["asr", "gold"], default="asr",
                   help="asr=餵 ASR 逐字稿(預設);gold=餵人工逐字稿,切開兩層誤差")
    r.add_argument("--only", help="只跑這幾個 id,逗號分隔")
    r.add_argument("--include-unusable", action="store_true",
                   help="連沒有對應 input 的 item 也跑(預設跳過)")
    r.add_argument("--notes")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="對既有 run 重算分數(不重跑模型)")
    s.add_argument("run_id")
    s.add_argument("--dataset")
    s.add_argument("--n-cjk", type=int)
    s.add_argument("--n-latin", type=int)
    s.set_defaults(fn=cmd_score)

    j = sub.add_parser("judge", help="對既有 run 補跑語意漂移判定")
    j.add_argument("run_id")
    j.set_defaults(fn=cmd_judge)

    rp = sub.add_parser("report", help="跨 run 對照表")
    rp.add_argument("--arm", choices=["polish", "asr", "e2e"])
    rp.add_argument("--dataset")
    rp.add_argument("--since", help="YYYY-MM-DD")
    rp.add_argument("--label", help="label 含這個字串的才列")
    rp.set_defaults(fn=cmd_report)

    sh = sub.add_parser("show", help="看單一 run/clip 的輸入輸出與標記")
    sh.add_argument("run_id")
    sh.add_argument("item", nargs="?")
    sh.set_defaults(fn=cmd_show)

    ls = sub.add_parser("list", help="列出所有 run")
    ls.set_defaults(fn=cmd_list)

    lg = sub.add_parser("legacy", help="把 out/ 的歷史輸出匯成 run")
    lg.add_argument("--which", choices=["asr", "polish", "all"], default="all")
    lg.set_defaults(fn=cmd_legacy)

    ri = sub.add_parser("reindex", help="從 runs/*/run.json 重建 index.jsonl")
    ri.set_defaults(fn=cmd_reindex)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
