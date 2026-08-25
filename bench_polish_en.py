#!/usr/bin/env python3
"""English control for the polish benchmark. Same task, same prompt structure,
same clips (translated 1:1 preserving disfluency), English instead of zh-TW.

  ./bench_polish_en.py --url http://localhost:8900 --model bonsai-27b-1bit --tag bonsai
"""
import argparse, json, pathlib, re, time, urllib.request

ROOT = pathlib.Path(__file__).parent
PROMPT, TEMP = None, 0.0

# tier-A vocalizations the prompt says to DELETE EVERY OCCURRENCE of
TIER_A = re.compile(r"\b(um+|uh+|er|erm|mmm+|hmm+|ah)\b", re.I)
# immediately repeated word (>=2 in a row), the "collapse stutters" rule
STUTTER = re.compile(r"\b(\w+)\b(?:[,\s]+\b\1\b)+", re.I)


def nonascii(s):
    return [c for c in s if ord(c) > 127 and c not in "‘’“”—–…é"]


def call(raw, url, model, timeout=2400):
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
    a.add_argument("--url", default="http://localhost:8900")
    a.add_argument("--model", default="bonsai-27b-1bit")
    a.add_argument("--tag", default="bonsai")
    a.add_argument("--prompt", default="prompts/cleanup-en.txt")
    a.add_argument("--temp", type=float, default=0.0)
    o = a.parse_args()
    global PROMPT, TEMP
    TEMP = o.temp
    PROMPT = (ROOT / o.prompt).read_text().replace("{{agentName}}", "assistant")

    call("warm up.", o.url, o.model)          # prime the system-prompt cache
    rows = []
    for k in ["16", "17", "14", "18"]:
        raw = (ROOT / "out" / f"en_{k}_raw.txt").read_text().strip()
        txt, el, t, u = call(raw, o.url, o.model)
        (ROOT / "out" / f"en_{k}_polished_{o.tag}.txt").write_text(txt + "\n")
        fa_r, fa_p = len(TIER_A.findall(raw)), len(TIER_A.findall(txt))
        st_r, st_p = len(STUTTER.findall(raw)), len(STUTTER.findall(txt))
        na = len(nonascii(txt))
        rows.append((k, el, u.get("completion_tokens", 0), t.get("predicted_per_second", 0),
                     fa_r, fa_p, st_r, st_p, na, len(txt) / max(len(raw), 1)))
        print(f"\n{'='*78}\n### en_{k}\nRAW    : {raw}\n\nPOLISH : {txt}")
        print(f"\n[{el:.1f}s | gen {u.get('completion_tokens')} @ {t.get('predicted_per_second',0):.1f} tok/s "
              f"| tier-A {fa_r}->{fa_p} | stutter {st_r}->{st_p} | non-ASCII {na} | len {len(txt)/max(len(raw),1):.2f}]")

    print(f"\n\n=== {o.tag} (english control, temp {TEMP}) ===")
    print(f"{'clip':<6}{'s':>8}{'gen':>6}{'tok/s':>8}{'tierA':>10}{'stutter':>10}{'nonASCII':>10}{'len':>7}")
    for k, el, g, tp, far, fap, str_, stp, na, lr in rows:
        print(f"{k:<6}{el:>8.1f}{g:>6}{tp:>8.1f}{f'{far}->{fap}':>10}{f'{str_}->{stp}':>10}{na:>10}{lr:>7.2f}")
    n = len(rows)
    print(f"\n平均 decode {sum(r[3] for r in rows)/n:.1f} tok/s | tier-A 殘留合計 {sum(r[5] for r in rows)}"
          f" (raw {sum(r[4] for r in rows)}) | stutter 殘留合計 {sum(r[7] for r in rows)} (raw {sum(r[6] for r in rows)})"
          f" | non-ASCII 合計 {sum(r[8] for r in rows)}")


if __name__ == "__main__":
    main()
