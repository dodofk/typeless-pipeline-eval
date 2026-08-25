#!/usr/bin/env python3
"""
從 OpenWhispr 導出 eval corpus。

  SQLite (transcriptions) + audio/*.webm  ->  corpus/*.wav + corpus.jsonl

音檔是 .webm，whisper 要 16kHz mono WAV，所以順便用 ffmpeg 轉檔。
失敗/取消的那些 (status != 'completed') 預設也會導出——**那些正是最該進 eval set 的樣本**。

usage:
    ./export_corpus.py                      # 全部
    ./export_corpus.py --only-failed        # 只要翻車的
    ./export_corpus.py --min-ms 3000        # 濾掉太短的
"""
import argparse, json, pathlib, sqlite3, subprocess, sys

DEV = pathlib.Path.home() / "Library/Application Support/OpenWhispr-development"
PROD = pathlib.Path.home() / "Library/Application Support/OpenWhispr"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app-dir", type=pathlib.Path,
                   help="OpenWhispr userData 目錄（預設自動找 dev 再找 prod）")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("corpus"))
    p.add_argument("--only-failed", action="store_true", help="只導出 status != completed 的")
    p.add_argument("--min-ms", type=int, default=0, help="濾掉短於這個毫秒數的")
    a = p.parse_args()

    root = a.app_dir or next((d for d in (DEV, PROD) if d.exists()), None)
    if not root:
        sys.exit(f"找不到 OpenWhispr 資料夾，試過：\n  {DEV}\n  {PROD}")
    db = next(root.glob("transcriptions*.db"), None)
    if not db:
        sys.exit(f"{root} 裡沒有 transcriptions*.db")

    a.out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    sql = ("select id, text, raw_text, timestamp, audio_duration_ms, provider, model, "
           "status, error_message, has_audio from transcriptions "
           "where deleted_at is null and has_audio = 1")
    if a.only_failed:
        sql += " and (status is null or status != 'completed')"
    if a.min_ms:
        sql += f" and coalesce(audio_duration_ms, 0) >= {a.min_ms}"

    kept = skipped = 0
    with (a.out.parent / "corpus.jsonl").open("w") as fh:
        for r in con.execute(sql + " order by timestamp"):
            # 檔名格式：OpenWhispr-YYYY-MM-DD-HH-MM-SS-<id>.webm — id 在最後，
            # 不能用 *<id>*.webm（日期裡的數字會亂比對）
            src = next((root / "audio").glob(f"*-{r['id']}.webm"), None)
            if not src:
                skipped += 1
                continue
            wav = a.out / f"{r['id']}.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000",
                            "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                           capture_output=True, check=True)
            json.dump({"id": r["id"], "wav": str(wav), "webm": str(src),
                       "asr_raw": r["raw_text"], "app_polished": r["text"],
                       "duration_ms": r["audio_duration_ms"], "provider": r["provider"],
                       "model": r["model"], "status": r["status"],
                       "error": r["error_message"], "timestamp": r["timestamp"]},
                      fh, ensure_ascii=False)
            fh.write("\n")
            kept += 1

    print(f"導出 {kept} 段 → {a.out}/ + corpus.jsonl", file=sys.stderr)
    if skipped:
        print(f"⚠️  {skipped} 筆有 has_audio=1 但找不到 .webm（可能已過 30 天 retention）", file=sys.stderr)


if __name__ == "__main__":
    main()
