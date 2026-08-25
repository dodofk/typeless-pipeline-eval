#!/usr/bin/env python3
"""polish.py —— 跑潤稿 LLM,產出 grade.py 吃得下的 JSONL。

    uv run python polish.py --asr asr/breeze.jsonl --out polish/breeze-qwen35-v2.jsonl

輸入就是 asr.py 的輸出(或任何符合 --asr 格式的檔案)。
輸出每一筆都帶 `input`,所以 grade.py 只吃 --polish 也算得出移除率和幻覺率。

模型不是這支起的 —— llama-server 要自己先跑起來(`./run_qwen35.sh`)。
這支只負責送文字、收結果、記 timing。

⚠️ 速度數字只有在**獨佔機器**時可比(坑 #7)。開頭會檢查有幾個 llama-server
在跑,>1 就在輸出裡標記 speed_trustworthy=false —— 標記而不是拒跑,
因為 CER 那些指標不受影響,只有 tok/s 不能比。

逐筆 append,中途掛掉不會全白跑;重跑自動跳過已完成的 id(--overwrite 可覆蓋)。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from typeless.engines import PolishConfig, llama_server_count, load_prompt, polish, warm  # noqa: E402


def read_jsonl(p: pathlib.Path) -> list[dict]:
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{p}:{n} 不是合法 JSON —— {e}")
    return out


def done_ids(out: pathlib.Path) -> set[str]:
    if not out.exists():
        return set()
    ids = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="跑潤稿,吐 grade.py 的 --polish 格式",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--asr", required=True, help="輸入 JSONL(asr.py 的輸出)")
    ap.add_argument("--out", required=True, help="輸出 JSONL")
    ap.add_argument("--url", default="http://127.0.0.1:8902", help="llama-server 位址")
    ap.add_argument("--model", default="qwen3.5-4b", help="只是記在輸出裡,不影響送出的請求")
    ap.add_argument("--prompt", default="prompts/cleanup-zhTW-mixed-v2.txt")
    ap.add_argument("--temp", type=float, default=0.0, help="A/B 比較一定要 0(坑 #3)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--only", help="只跑這些 id,逗號分隔")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-warm", action="store_true", help="跳過暖機(第一筆會多付一次 prefill)")
    a = ap.parse_args()

    src = pathlib.Path(a.asr)
    if not src.exists():
        raise SystemExit(f"--asr 不存在:{src}")
    recs = read_jsonl(src)
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        recs = [r for r in recs if r.get("id") in want]

    # 上游失敗的 item 不送 —— 空字串送進去只會拿到一段幻覺
    skipped_empty = [r["id"] for r in recs if not (r.get("text") or "").strip()]
    recs = [r for r in recs if (r.get("text") or "").strip()]
    if skipped_empty:
        print(f"⚠️ 跳過 {len(skipped_empty)} 筆上游是空的:{', '.join(skipped_empty)}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = set() if a.overwrite else done_ids(out)
    if a.overwrite and out.exists():
        out.unlink()
    todo = [r for r in recs if r["id"] not in skip]
    if skip:
        print(f"跳過 {len(skip)} 個已完成的 id(--overwrite 可重跑)")
    if not todo:
        print("沒有要跑的。")
        return 0

    prompt, sha = load_prompt(a.prompt)
    cfg = PolishConfig(url=a.url, prompt_file=a.prompt, temp=a.temp, seed=a.seed,
                       max_tokens=a.max_tokens, prompt_sha=sha)
    nsrv = llama_server_count()
    trust = nsrv == 1
    print(f"模型 {a.model}  prompt {pathlib.Path(a.prompt).name}#{sha}  temp {a.temp}")
    print(f"llama-server 數量 {nsrv}" + ("" if trust else "  ⚠️ 速度數字不可比"))
    print(f"{len(todo)} 筆  →  {out}\n")

    if not a.no_warm:
        print("暖機…", flush=True)
        warm(cfg, prompt)

    ok = err = 0
    t_all = time.monotonic()
    with out.open("a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            print(f"[{n}/{len(todo)}] {r['id']:<12}", end="", flush=True)
            res = polish(r["text"], cfg, prompt)
            rec = {"id": r["id"], "text": res.text, "input": r["text"],
                   "dur_s": r.get("dur_s"), "model": a.model,
                   "prompt_file": a.prompt, "prompt_sha": sha,
                   "temp": a.temp, "seed": a.seed,
                   "speed_trustworthy": trust, "llama_servers": nsrv}
            if res.error:
                err += 1
                rec["error"] = res.error
                print(f"  ✗ {res.error[:90]}")
            else:
                ok += 1
                t = res.timings
                rec.update({"latency_s": round(t.get("wall_s") or 0, 3),
                            "gen_tok": t.get("gen_tok"), "tok_s": t.get("tok_s"),
                            "prompt_tok": t.get("prompt_tok")})
                ratio = len(res.text) / len(r["text"]) if r["text"] else 0
                print(f"  {rec['latency_s']:6.1f}s  {rec['gen_tok'] or 0:>5} tok"
                      f"  @{(rec['tok_s'] or 0):5.1f} tok/s  長度比 {ratio:.2f}"
                      + ("  ⚠️刪過頭?" if ratio < 0.7 else ""))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n完成 {ok} 失敗 {err}  總共 {time.monotonic() - t_all:.1f}s  →  {out}")
    if not trust:
        print("⚠️ 機器上有多個 llama-server,tok/s 不可跨 run 比較。")
    return 1 if err and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
