#!/usr/bin/env python3
"""Paired EN/ZH capability probes: is the 1-bit damage language-specific or global?

Each probe has an English and a semantically identical Chinese version. Same model,
same sampling. If EN passes and ZH fails -> language-specific. If both fail -> global.

  ./probe_lang.py --url http://localhost:8900 --model bonsai-27b-1bit --tag bonsai
"""
import argparse, json, re, time, urllib.request

ECHO_EN = ("The rerank layer adds 42 ms of latency per query, which is a real trade-off "
           "when the p99 budget is 300 ms, so we cache the top-K candidates.")
ECHO_ZH = "重排層每個查詢會增加 42 毫秒的延遲，在 p99 預算只有 300 毫秒時這是實際的取捨，所以我們把 top-K 候選快取起來。"

PROBES = [
    # (id, kind, english, chinese)
    ("echo",  "fidelity",
     f"Repeat the following text back exactly, changing nothing. Output only the text.\n\n{ECHO_EN}",
     f"請把下面這段文字原封不動地重複一次，一個字都不要改。只輸出這段文字。\n\n{ECHO_ZH}"),
    ("math",  "structured",
     "One batch takes 3.5 minutes to process. There are 48 batches. How many hours in total? "
     "Answer with just the number of hours.",
     "處理一個 batch 要 3.5 分鐘，總共有 48 個 batch。全部要幾個小時？只回答小時數。"),
    ("code",  "structured",
     "Write a Python function second_largest(nums) returning the second largest DISTINCT number, "
     "or None if there isn't one. Output only the code.",
     "寫一個 Python 函式 second_largest(nums)，回傳第二大的「相異」數字，沒有的話回傳 None。只輸出程式碼。"),
    ("fmt",   "instruction",
     "List exactly three risks of 4-bit quantization. One per line. Each under 10 words. "
     "No numbering, no bullets, no preamble.",
     "列出 4-bit 量化的三個風險，剛好三個。一行一個。每行不超過 15 個字。"
     "不要編號、不要項目符號、不要開場白。"),
    ("gen",   "generation",
     "Explain in exactly 3 sentences why memory bandwidth limits LLM decoding speed.",
     "用剛好 3 句話解釋為什麼記憶體頻寬會限制 LLM 的解碼速度。"),
]


def call(prompt, url, model, temp, timeout=2400):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp, "top_p": 0.95, "top_k": 20, "max_tokens": 768, "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(
        urllib.request.Request(f"{url}/v1/chat/completions", body,
                               {"Content-Type": "application/json"}), timeout=timeout))
    txt = re.sub(r"^\s*<think>.*?</think>\s*", "", d["choices"][0]["message"]["content"], flags=re.S).strip()
    return txt, time.time() - t0, d.get("usage", {})


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://localhost:8900")
    a.add_argument("--model", default="bonsai-27b-1bit")
    a.add_argument("--tag", default="bonsai")
    a.add_argument("--temp", type=float, default=0.0)
    o = a.parse_args()

    call("hi", o.url, o.model, o.temp)  # warm
    for pid, kind, en, zh in PROBES:
        for lang, prompt in (("EN", en), ("ZH", zh)):
            txt, el, u = call(prompt, o.url, o.model, o.temp)
            open(f"out/probe_{pid}_{lang}_{o.tag}.txt", "w").write(txt + "\n")
            print(f"\n{'='*78}\n### [{o.tag}] {pid} / {lang} ({kind})  {el:.1f}s  "
                  f"{u.get('completion_tokens')} tok\n{'-'*78}\n{txt}", flush=True)


if __name__ == "__main__":
    main()
