"""ASR 輸出的來源 —— 被測的那一端。

架構上要分清楚兩件事,弄反的話就是拿答案當考卷:

  ground truth ← Spokenly + ElevenLabs + 人工修正 → evalset/text/<id>.zh-tw.txt
  hypothesis   ← OpenWhispr 的 audio upload,抽換不同 ASR 模型 → 這裡

同一個 evalset item 會有 N 份 hypothesis(一個 ASR 模型一份),全部對同一份
ground truth 計分。所以 raw 不屬於 dataset,屬於 run —— 由 --asr-source 指定。

--- OpenWhispr 的資料落點 ---

audio upload 走 saveUploadNote() → notes 表:
    source_file             上傳時的原始檔名 ← 跟 evalset id 的接合鍵
    content                 ASR 逐字稿
    enhanced_content        OpenWhispr 自己的 AI 潤稿(可以拿來當對照組)
    audio_duration_seconds
    folder_id               ← 分群用
    note_type = 'upload'

⚠️ notes 表**沒有 provider/model 欄位** —— 上傳結果不記錄是哪個 ASR 跑的。
   同樣的音檔換三個模型各傳一次,三批 note 在 DB 裡分不出來。
   解法:一個模型建一個 folder,各自傳進去,然後 --asr-source openwhispr:<folder>。
   退路:用 --since/--until 靠 created_at 切批次。
"""

from __future__ import annotations

import pathlib
import re
import sqlite3

DEV = pathlib.Path.home() / "Library/Application Support/OpenWhispr-development"
PROD = pathlib.Path.home() / "Library/Application Support/OpenWhispr"


def _stem(name: str) -> str:
    """agent-01.wav → agent-01。OpenWhispr 會原樣保留上傳的檔名。"""
    return pathlib.Path(str(name or "")).stem


def find_openwhispr_db(app_dir: pathlib.Path | str | None = None) -> pathlib.Path:
    roots = [pathlib.Path(app_dir).expanduser()] if app_dir else [DEV, PROD]
    for r in roots:
        if r.exists():
            db = next(r.glob("transcriptions*.db"), None)
            if db:
                return db
    raise SystemExit("找不到 OpenWhispr 的 DB,試過:\n  " + "\n  ".join(str(r) for r in roots))


def from_openwhispr(folder: str | None = None, since: str | None = None,
                    until: str | None = None, field: str = "content",
                    app_dir=None) -> tuple[dict, dict]:
    """回傳 ({evalset_id: 文字}, meta)。

    field: content=ASR 逐字稿(預設);enhanced_content=OpenWhispr 自己的潤稿。
    """
    db = find_openwhispr_db(app_dir)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    if field not in ("content", "enhanced_content", "transcript"):
        raise SystemExit(f"不認識的欄位:{field}")

    sql = (f"select id, title, source_file, {field} as text, audio_duration_seconds, "
           f"folder_id, created_at from notes "
           f"where note_type = 'upload' and deleted_at is null")
    params: list = []
    if folder:
        row = con.execute("select id from folders where name = ?", (folder,)).fetchone()
        if not row:
            names = [r["name"] for r in con.execute("select name from folders")]
            raise SystemExit(f"OpenWhispr 裡沒有叫 '{folder}' 的 folder。"
                             f"現有的:{', '.join(names)}")
        sql += " and folder_id = ?"
        params.append(row["id"])
    if since:
        sql += " and created_at >= ?"
        params.append(since)
    if until:
        sql += " and created_at <= ?"
        params.append(until)

    rows = list(con.execute(sql + " order by created_at", params))
    out, dupes = {}, []
    for r in rows:
        key = _stem(r["source_file"]) or f"note-{r['id']}"
        if key in out:
            dupes.append(key)        # 同一個檔傳了兩次 → 後傳的贏,但要講
        out[key] = (r["text"] or "").strip()

    meta = {"source": "openwhispr", "db": str(db), "folder": folder,
            "field": field, "since": since, "until": until,
            "n_notes": len(rows), "duplicates": sorted(set(dupes))}
    return out, meta


def from_spokenly(evalset_root: pathlib.Path | str, stage: str = "original") -> tuple[dict, dict]:
    """Spokenly 匯出的某個 stage。

    ⚠️ 這是 ground truth 的來源引擎(ElevenLabs)。拿它當 hypothesis 等於用
    答案當考卷 —— 分數會是天花板不是實力。只有在刻意要量上界時才用。
    """
    from .dataset import spokenly_stages
    root = pathlib.Path(evalset_root).expanduser()
    out = {}
    for js in sorted((root / "raw").glob("*.json")):
        st = spokenly_stages(js)
        if st.get(stage):
            out[js.stem] = st[stage]
    return out, {"source": "spokenly", "stage": stage, "root": str(root),
                 "warning": "這是 ground truth 的來源引擎,分數是上界不是實力"}


def from_dir(path: pathlib.Path | str) -> tuple[dict, dict]:
    """一個目錄,裡面是 <evalset_id>.txt。任何 ASR 的逃生出口。"""
    p = pathlib.Path(path).expanduser()
    if not p.is_dir():
        raise SystemExit(f"不是目錄:{p}")
    out = {f.stem: f.read_text().strip() for f in sorted(p.glob("*.txt"))}
    return out, {"source": "dir", "path": str(p), "n_files": len(out)}


def resolve(spec: str, evalset_root=None, **kw) -> tuple[dict, dict]:
    """--asr-source 的字串 → (文字對照表, meta)。

        openwhispr                OpenWhispr 全部的 upload note
        openwhispr:whisper-turbo  只要那個 folder 的
        openwhispr-polished       OpenWhispr 自己的 AI 潤稿(對照組)
        spokenly                  Spokenly 的 ASR 原文(= GT 引擎,上界)
        spokenly:smartParagraphs  Spokenly 的某個後處理 stage
        dir:path/to/txts          一個目錄的 <id>.txt
        path/to/txts              同上,省略前綴
    """
    name, _, arg = spec.partition(":")
    if name == "openwhispr":
        return from_openwhispr(folder=arg or None, **kw)
    if name == "openwhispr-polished":
        return from_openwhispr(folder=arg or None, field="enhanced_content", **kw)
    if name == "spokenly":
        if not evalset_root:
            raise SystemExit("spokenly 來源需要知道 evalset 的路徑")
        return from_spokenly(evalset_root, stage=arg or "original")
    if name == "dir":
        return from_dir(arg)
    if pathlib.Path(spec).expanduser().is_dir():
        return from_dir(spec)
    raise SystemExit(f"不認識的 --asr-source:{spec}\n可用:openwhispr[:folder]、"
                     f"openwhispr-polished、spokenly[:stage]、dir:<path>")
