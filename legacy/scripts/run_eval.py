#!/usr/bin/env python3
"""
local-typeless eval loop.

    .wav -> whisper-cli -> raw text -> llama-server -> polished text

不碰 app。zero dependency（只用 stdlib）。
換模型 = 重啟 llama-server 換 -m 參數，這支程式一行都不用改。

usage:
    # Mode B（整段送，可做 bullet 結構化）
    ./run_eval.py --audio audio/ --label gemma4-e4b-q4 --mode full

    # Mode A（一句一句送，模擬即時輸入）
    ./run_eval.py --audio audio/ --label gemma4-e4b-q4 --mode sentence

    # gold-input arm：跳過 STT，直接餵人工逐字稿（切開兩層誤差用）
    ./run_eval.py --gold gold/ --label gemma4-e4b-q4 --mode full
"""
import argparse, json, pathlib, subprocess, sys, time, urllib.request

# TODO(D2): 換成從 llama-server verbose log 撈出來的 OpenWhispr production prompt
POLISH_PROMPT = """你是一個語音輸入的後處理器。使用者的語音被轉成逐字稿，可能有口語冗詞、\
標點缺失、同音錯字、中英混用。請輸出可以直接送出的文字。

規則：
- 去掉「呃」「就是」「然後那個」這類填充詞與重複
- 補正標點與分段
- 修正明顯的同音錯字，但**不要**新增原文沒有的資訊
- 英文術語、專有名詞保留原文，不要翻譯
- 只輸出處理後的文字，不要加任何說明"""


def transcribe(wav, model, whisper_bin, lang, initial_prompt=None, vad=False):
    """回傳 (full_text, [segment_texts], elapsed_sec)"""
    out = wav.with_suffix("")
    cmd = [whisper_bin, "-m", model, "-f", str(wav), "-oj", "-of", str(out), "-l", lang]
    if initial_prompt:
        cmd += ["--prompt", initial_prompt]
    if vad:
        cmd += ["--vad"]          # 對齊 OpenWhispr；注意他們預設是關的
    t0 = time.monotonic()
    subprocess.run(cmd, check=True, capture_output=True)
    elapsed = time.monotonic() - t0
    data = json.loads(out.with_suffix(".json").read_text())
    segs = [s["text"].strip() for s in data["transcription"] if s["text"].strip()]
    return " ".join(segs), segs, elapsed


def polish(text, url, prompt):
    """回傳 (polished_text, timings_dict)。timings 是 llama-server 免費送的。"""
    body = json.dumps({
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": text}],
        "temperature": 0.2, "stream": False,
        # ⚠️ 關鍵：gemma-4 預設開 thinking，一句話 polish 會噴 761 tokens / 19.4s。
        # 關掉後 21 tokens / 0.50s——差 39 倍。request 裡的 "reasoning_budget":0 無效。
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r["choices"][0]["message"]["content"].strip(), {
        "wall_sec": round(time.monotonic() - t0, 3), **r.get("timings", {})}


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=pathlib.Path, help="放 .wav 的目錄（跑完整 pipeline）")
    src.add_argument("--gold", type=pathlib.Path, help="放 .txt 人工逐字稿的目錄（gold-input arm）")
    p.add_argument("--label", required=True, help="這個 config 的名字，會寫進結果")
    p.add_argument("--mode", choices=["full", "sentence"], default="full")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results.jsonl"))
    p.add_argument("--llama-url", default="http://127.0.0.1:8080")
    p.add_argument("--whisper-bin", default="whisper-cli")
    p.add_argument("--whisper-model", default="models/ggml-large-v3-turbo-q5_0.bin")
    p.add_argument("--lang", default="zh")
    p.add_argument("--initial-prompt", help="whisper 的 dictionary 注入點（小心 prompt echo）")
    p.add_argument("--vad", action="store_true")
    a = p.parse_args()

    files = sorted((a.audio or a.gold).glob("*.wav" if a.audio else "*.txt"))
    if not files:
        sys.exit(f"找不到檔案：{a.audio or a.gold}")

    with a.out.open("a") as fh:
        for i, f in enumerate(files, 1):
            if a.audio:
                raw, segs, stt_sec = transcribe(f, a.whisper_model, a.whisper_bin,
                                                a.lang, a.initial_prompt, a.vad)
                arm = "asr"
            else:
                raw = f.read_text().strip()
                segs, stt_sec, arm = [s for s in raw.split("\n") if s.strip()], None, "gold"

            units = segs if a.mode == "sentence" else [raw]
            outs, tims = [], []
            for u in units:
                o, t = polish(u, a.llama_url, POLISH_PROMPT)
                outs.append(o); tims.append(t)

            json.dump({"id": f.stem, "label": a.label, "mode": a.mode, "arm": arm,
                       "raw": raw, "polished": "\n".join(outs),
                       "stt_sec": stt_sec, "timings": tims}, fh, ensure_ascii=False)
            fh.write("\n"); fh.flush()
            print(f"[{i}/{len(files)}] {f.stem}", file=sys.stderr)

    print(f"\n完成 → {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
