#!/usr/bin/env python3
"""real/*.wav -> Fun-ASR (:8379) -> Bonsai-27B-1bit polish (:8900). The full local pipeline.

  ./run_bonsai27b.sh &                    # polish LLM
  python3 funasr_server.py &              # STT
  ./e2e_real.py real/*.wav
"""
import argparse, json, mimetypes, pathlib, re, sys, time, urllib.request, uuid

ROOT = pathlib.Path(__file__).parent
PROMPT = (ROOT / "prompts" / "cleanup-zhTW-mixed.txt").read_text().replace("{{agentName}}", "assistant")
DICT = "OpenWhispr, baseline, latency, throughput, pipeline, embedding, rerank, commit, deploy, repo, push"


def stt(wav, url, model="funasr-q8_0"):
    """multipart POST to the OpenAI-compatible /v1/audio/transcriptions shim."""
    b = uuid.uuid4().hex
    body = b""
    for k, v in {"model": model, "response_format": "json", "prompt": DICT}.items():
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    data = wav.read_bytes()
    ctype = mimetypes.guess_type(wav.name)[0] or "audio/wav"
    body += (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{wav.name}\"\r\n"
             f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"{url}/v1/audio/transcriptions", body,
                                 {"Content-Type": f"multipart/form-data; boundary={b}"})
    t0 = time.time()
    return json.load(urllib.request.urlopen(req, timeout=1800))["text"].strip(), time.time() - t0


def polish(raw, url, model):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": f"<transcript>\n{raw}\n</transcript>"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},   # polish needs no reasoning trace
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    o = json.load(urllib.request.urlopen(req, timeout=1800))
    txt = re.sub(r"^\s*<think>.*?</think>\s*", "", o["choices"][0]["message"]["content"], flags=re.S).strip()
    return txt, time.time() - t0, o.get("usage", {}).get("completion_tokens", 0)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("wavs", nargs="+")
    a.add_argument("--stt-url", default="http://127.0.0.1:8379")
    a.add_argument("--llm-url", default="http://127.0.0.1:8900")
    a.add_argument("--llm-model", default="bonsai-27b-1bit")
    a.add_argument("--stt-model", default="funasr-q8_0")
    a.add_argument("--save", action="store_true")
    o = a.parse_args()

    for w in [pathlib.Path(x) for x in o.wavs]:
        print(f"\n{'='*80}\n### {w.name}")
        try:
            raw, t_stt = stt(w, o.stt_url, o.stt_model)
        except Exception as e:
            print(f"  STT FAILED: {type(e).__name__}: {e}"); continue
        print(f"\n[STT {t_stt:.1f}s]  {raw}")
        try:
            txt, t_llm, ntok = polish(raw, o.llm_url, o.llm_model)
        except Exception as e:
            print(f"  POLISH FAILED: {type(e).__name__}: {e}"); continue
        print(f"\n[POLISH {t_llm:.1f}s, {ntok} tok, {ntok/max(t_llm,1e-9):.1f} tok/s]\n{txt}")
        if o.save:
            p = ROOT / "out" / f"e2e_{w.stem}"
            (ROOT / "out").mkdir(exist_ok=True)
            pathlib.Path(f"{p}_raw.txt").write_text(raw + "\n")
            pathlib.Path(f"{p}_polished.txt").write_text(txt + "\n")
            print(f"-> out/e2e_{w.stem}_{{raw,polished}}.txt")


if __name__ == "__main__":
    main()
