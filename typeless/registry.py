"""run registry —— 每次跑都是一筆可累積、可比較、可回頭重現的記錄。

    runs/<run_id>/run.json         完整記錄(config + 每個 item + 總計)
    runs/<run_id>/items/<id>.txt   模型實際輸出的文字
    runs/index.jsonl               一行一 run,report 只讀這個(不用開 122 個檔)

最關鍵的設計:**跑模型和計分分離**。
run.json 一定存 `raw` 和 `out` 的原文,所以改了 metric 之後可以對所有歷史 run
重算分數,不用重起模型。舊腳本(bench_polish.py)是算完就印、數字不落地,
換一版 metric 就得把全部模型再跑一次 —— 那是這套東西要解決的主要痛點。
"""

from __future__ import annotations

import json
import pathlib
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
INDEX = RUNS / "index.jsonl"

_SLUG = re.compile(r"[^a-zA-Z0-9._-]+")


def make_run_id(label: str, when: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when or time.time()))
    return f"{stamp}-{_SLUG.sub('-', label).strip('-') or 'run'}"


def env_snapshot(exclusive_expected: bool = True) -> dict:
    """跑的當下環境長怎樣。速度數字可不可信全靠這個。"""
    from .engines import llama_server_count
    n = llama_server_count()
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        git = ""
    return {
        "llama_procs": n,
        # 坑 #7:16GB 機器,兩個 llama-server 會互搶記憶體頻寬。
        # 這個旗標一旦是 False,report 就把 tok/s 標成不可比。
        "speed_trustworthy": (n <= 1) if exclusive_expected else False,
        "git": git,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
    }


@dataclass
class Run:
    run_id: str
    label: str
    arm: str                       # polish | asr | e2e
    dataset: str
    config: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    items: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    created: str = ""
    notes: str = ""

    @property
    def dir(self) -> pathlib.Path:
        return RUNS / self.run_id

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "created": self.created, "label": self.label,
                "arm": self.arm, "dataset": self.dataset, "config": self.config,
                "env": self.env, "notes": self.notes,
                "aggregate": self.aggregate, "items": self.items}

    # ------------------------------------------------------------ 寫
    def save(self) -> pathlib.Path:
        self.created = self.created or time.strftime("%Y-%m-%dT%H:%M:%S%z")
        d = self.dir
        (d / "items").mkdir(parents=True, exist_ok=True)
        for it in self.items:
            if it.get("out"):
                (d / "items" / f"{it['id']}.txt").write_text(it["out"] + "\n")
        (d / "run.json").write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2))
        self._reindex()
        return d / "run.json"

    def _index_row(self) -> dict:
        return {"run_id": self.run_id, "created": self.created, "label": self.label,
                "arm": self.arm, "dataset": self.dataset, "n": len(self.items),
                "model": (self.config.get("polish") or {}).get("model"),
                "prompt": (self.config.get("polish") or {}).get("prompt_file"),
                "prompt_sha": (self.config.get("polish") or {}).get("prompt_sha"),
                "temp": (self.config.get("polish") or {}).get("temp"),
                "input": self.config.get("input"),
                "speed_trustworthy": self.env.get("speed_trustworthy"),
                "aggregate": self.aggregate}

    def _reindex(self) -> None:
        RUNS.mkdir(exist_ok=True)
        rows = [r for r in read_index() if r.get("run_id") != self.run_id]
        rows.append(self._index_row())
        rows.sort(key=lambda r: r.get("created") or "")
        with INDEX.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ 讀
    @classmethod
    def load(cls, run_id: str) -> "Run":
        p = RUNS / run_id / "run.json"
        if not p.exists():
            matches = [d.name for d in RUNS.glob(f"*{run_id}*") if (d / "run.json").exists()] \
                if RUNS.exists() else []
            if len(matches) == 1:
                p = RUNS / matches[0] / "run.json"
            elif matches:
                raise SystemExit(f"'{run_id}' 對到多個 run:\n  " + "\n  ".join(matches))
            else:
                raise SystemExit(f"找不到 run:{run_id}  (看看 ./tl list)")
        d = json.loads(p.read_text())
        return cls(run_id=d["run_id"], label=d["label"], arm=d["arm"],
                   dataset=d["dataset"], config=d.get("config", {}),
                   env=d.get("env", {}), items=d.get("items", []),
                   aggregate=d.get("aggregate", {}), created=d.get("created", ""),
                   notes=d.get("notes", ""))


def read_index() -> list[dict]:
    if not INDEX.exists():
        return []
    rows = []
    for line in INDEX.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def rebuild_index() -> int:
    """index.jsonl 掉了或壞了,從 runs/*/run.json 重建。"""
    RUNS.mkdir(exist_ok=True)
    rows = []
    for p in sorted(RUNS.glob("*/run.json")):
        r = Run.load(p.parent.name)
        rows.append(r._index_row())
    rows.sort(key=lambda r: r.get("created") or "")
    with INDEX.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
