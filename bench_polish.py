#!/usr/bin/env python3
"""Same prompt, same transcripts, any llama-server. Reports latency + 繁簡 residue.

  ./bench_polish.py --url http://localhost:8901 --model ornith-1.5-9b --tag ornith
"""
import argparse, json, pathlib, re, time, urllib.request
try: import zhconv
except ImportError: raise SystemExit("pip3 install zhconv")

ROOT = pathlib.Path(__file__).parent
PROMPT = None  # set in main()
TEMP = 0.7
DUR = {"14": 69.0, "16": 28.1, "17": 33.1, "18": 76.4}
FILLERS = ["呃", "嗯", "就是說", "對對對", "那個那個"]


def simp(s):
    return [c for c in s if c != zhconv.convert(c, "zh-tw")]


def call(raw, url, model, timeout=1800):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": f"<transcript>\n{raw}\n</transcript>"}],
        "temperature": TEMP, "top_p": 0.95, "top_k": 20, "max_tokens": 1536, "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(
        urllib.request.Request(f"{url}/v1/chat/completions", body,
                               {"Content-Type": "application/json"}), timeout=timeout))
    txt = re.sub(r"^\s*<think>.*?</think>\s*", "", d["choices"][0]["message"]["content"], flags=re.S).strip()
    return txt, time.time() - t0, d.get("timings", {}), d.get("usage", {})


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://localhost:8901")
    a.add_argument("--model", default="ornith-1.5-9b")
    a.add_argument("--tag", default="ornith")
    a.add_argument("--prompt", default="prompts/cleanup-zhTW-mixed.txt")
    a.add_argument("--temp", type=float, default=0.7)
    o = a.parse_args()
    global PROMPT, TEMP
    TEMP = o.temp
    PROMPT = (ROOT / o.prompt).read_text().replace("{{agentName}}", "assistant")

    call("暖機。", o.url, o.model)          # prime the system-prompt cache
    rows = []
    for k in ["16", "17", "14", "18"]:
        raw = (ROOT / "out" / f"e2e_{k}_raw.txt").read_text().strip()
        txt, el, t, u = call(raw, o.url, o.model)
        (ROOT / "out" / f"e2e_{k}_polished_{o.tag}.txt").write_text(txt + "\n")
        sr, sp = len(simp(raw)), len(simp(txt))
        fr = sum(txt.count(f) for f in FILLERS)
        rows.append((k, DUR[k], el, u.get("completion_tokens", 0),
                     t.get("predicted_per_second", 0), sr, sp, fr))
        print(f"\n{'='*78}\n### {k}.wav ({DUR[k]}s)\nRAW    : {raw}\nPOLISH : {txt}")
        print(f"[{el:.1f}s | gen {u.get('completion_tokens')} @ {t.get('predicted_per_second',0):.1f} tok/s "
              f"| 簡體 {sr}->{sp} | filler殘留 {fr}]")

    print(f"\n\n=== {o.tag} ===")
    print(f"{'clip':<6}{'音檔s':>7}{'polish s':>10}{'gen':>6}{'tok/s':>8}{'簡體殘留':>10}{'filler':>8}")
    for k, d, el, g, sp_, sr, spo, fr in rows:
        print(f"{k:<6}{d:>7.1f}{el:>10.1f}{g:>6}{sp_:>8.1f}{f'{sr}->{spo}':>10}{fr:>8}")
    avg = sum(r[4] for r in rows) / len(rows)
    print(f"\n平均 decode {avg:.1f} tok/s | 簡體殘留合計 {sum(r[6] for r in rows)} | filler殘留合計 {sum(r[7] for r in rows)}")


if __name__ == "__main__":
    main()
