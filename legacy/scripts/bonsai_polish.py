#!/usr/bin/env python3
"""Run the zh-TW cleanup prompt through local Bonsai-27B-1bit (mlx_lm server) and compare to gold.

  ./bonsai_polish.py out/funasr_mm28_long1_meeting_f10.txt   # one file
  ./bonsai_polish.py --all                                   # every funasr_mm28_* clip
"""
import argparse, json, pathlib, re, sys, time, urllib.request

ROOT = pathlib.Path(__file__).parent
PROMPT = (ROOT / "prompts" / "cleanup-zhTW-mixed.txt").read_text().replace("{{agentName}}", "assistant")


def polish(raw, url, model, timeout=600):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"<transcript>\n{raw.strip()}\n</transcript>"},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    txt = out["choices"][0]["message"]["content"].strip()
    txt = re.sub(r"^<think>.*?</think>\s*", "", txt, flags=re.S)  # Qwen3 reasoning traces
    usage = out.get("usage", {})
    return txt, time.time() - t0, usage.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--url", default="http://localhost:8900")
    ap.add_argument("--model", default="bonsai-27b-1bit")
    ap.add_argument("--save", action="store_true", help="write to out/polished_<stem>.txt")
    a = ap.parse_args()

    files = sorted((ROOT / "out").glob("funasr_mm28_*_f10.txt")) if a.all \
        else [pathlib.Path(f) for f in a.files]
    if not files:
        sys.exit("no input files; pass paths or --all")

    for p in files:
        raw = p.read_text()
        txt, dt, ntok = polish(raw, a.url, a.model)
        print(f"\n{'='*78}\n### {p.name}   [{dt:.1f}s, {ntok} tok, {ntok/max(dt,1e-9):.1f} tok/s]\n{'-'*78}")
        print(f"RAW    : {raw.strip()}\n")
        print(f"POLISH : {txt}\n")
        stem = re.sub(r"^funasr_mm28_|_f10$|_m115$", "", p.stem)
        gold = ROOT / "gold" / f"{stem}.txt"
        if gold.exists():
            print(f"GOLD   : {gold.read_text().strip()}")
        if a.save:
            o = ROOT / "out" / f"polished_bonsai27b1bit_{p.stem}.txt"
            o.write_text(txt + "\n")
            print(f"\n-> {o}")


if __name__ == "__main__":
    main()
