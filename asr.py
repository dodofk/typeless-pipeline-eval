#!/usr/bin/env python3
"""asr.py —— 跑 ASR,產出 grade.py 吃得下的 JSONL。

    uv run python asr.py --engine whisper     --out asr/whisper-turbo.jsonl
    uv run python asr.py --engine funasr:q4km --out asr/funasr-q4km.jsonl
    uv run python asr.py --engine breeze      --out asr/breeze.jsonl
    uv run python asr.py --engine scribe      --out asr/scribe.jsonl

⚠️ 這條路跟走 OpenWhispr **不保證等價**,差在哪查過原始碼(見 typeless/asr_engines.py):

    funasr / breeze   OpenWhispr 走 self-hosted,只是把整個檔 POST 到你的 server,
                      沒有本機 VAD、沒有切段 → 等價,直接用。
    scribe            BYOK 直送 ElevenLabs → 等價。但它是 ground truth 的來源引擎,
                      CER 量到的是「人工當初改了多少」,是下界不是實力。
    whisper           ⚠️ 不保證。OpenWhispr 上傳走 "noteRecording" context,本機 VAD
                      預設是**開**的,而且那組參數跟 whisper-cli 自己的預設有 4 個不同。
                      這裡已經複製了 OpenWhispr 那組值,但要拿 app 跑一次對照過才能信。

寫檔是逐筆 append,中途掛掉不會全部白跑;再跑一次會自動跳過已完成的 id(--overwrite 可覆蓋)。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from typeless import asr_engines as E      # noqa: E402


def load_manifest(p: pathlib.Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            out[m["id"]] = m
    return out


def probe_dur(wav: pathlib.Path) -> float | None:
    """manifest 沒寫 duration 時才用 ffprobe 問。沒有 ffprobe 就回 None。"""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(wav)], capture_output=True, timeout=30)
        return round(float(r.stdout.decode().strip()), 2)
    except Exception:
        return None


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
    ap = argparse.ArgumentParser(description="跑 ASR,吐 grade.py 的 --asr 格式",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--engine", required=True,
                    help="whisper / whisper:no-vad / funasr:q4km / breeze / scribe")
    ap.add_argument("--audio", default="evalset/audio", help="音檔資料夾(預設 evalset/audio)")
    ap.add_argument("--manifest", default="evalset/manifest.jsonl",
                    help="拿 duration_sec 用;沒有就 ffprobe")
    ap.add_argument("--out", required=True, help="輸出 JSONL")
    ap.add_argument("--only", help="只跑這些 id,逗號分隔")
    ap.add_argument("--overwrite", action="store_true", help="重跑已完成的 id")
    a = ap.parse_args()

    audio = pathlib.Path(a.audio)
    if not audio.is_dir():
        raise SystemExit(f"--audio 不是資料夾:{audio}")
    wavs = sorted(audio.glob("*.wav"))
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        missing = want - {w.stem for w in wavs}
        if missing:
            raise SystemExit(f"--only 裡這些找不到音檔:{', '.join(sorted(missing))}")
        wavs = [w for w in wavs if w.stem in want]
    if not wavs:
        raise SystemExit(f"{audio} 裡沒有 wav")

    man = load_manifest(pathlib.Path(a.manifest))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = set() if a.overwrite else done_ids(out)
    if a.overwrite and out.exists():
        out.unlink()

    todo = [w for w in wavs if w.stem not in skip]
    if skip:
        print(f"跳過 {len(skip)} 個已完成的 id(--overwrite 可重跑)")
    if not todo:
        print("沒有要跑的。")
        return 0

    print(f"引擎 {a.engine}  |  {len(todo)} 個檔  →  {out}\n")
    ok = err = 0
    t_all = time.monotonic()
    with out.open("a", encoding="utf-8") as f:
        for n, w in enumerate(todo, 1):
            m = man.get(w.stem, {})
            dur = m.get("duration_sec") or probe_dur(w)
            print(f"[{n}/{len(todo)}] {w.stem:<12}", end="", flush=True)
            r = E.run(a.engine, w)
            if r.error:
                err += 1
                print(f"  ✗ {r.error.splitlines()[0][:90]}")
                f.write(json.dumps({"id": w.stem, "text": "", "dur_s": dur,
                                    "error": r.error, "engine": a.engine},
                                   ensure_ascii=False) + "\n")
            else:
                ok += 1
                rtf = f"{r.elapsed_s / dur:.2f}" if dur else "—"
                print(f"  {r.elapsed_s:6.1f}s  RTF {rtf:>5}  {len(r.text):>5} 字")
                f.write(json.dumps({"id": w.stem, "text": r.text, "dur_s": dur,
                                    "elapsed_s": round(r.elapsed_s, 3),
                                    "engine": a.engine, "invocation": r.invocation},
                                   ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n完成 {ok} 失敗 {err}  總共 {time.monotonic() - t_all:.1f}s  →  {out}")
    if err:
        print("⚠️ 失敗的 item 也寫進檔案了(text 是空字串 + error 欄位)—— "
              "grade.py 會把它當 0 字的輸出,不要直接拿去比。")
    return 1 if err and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
