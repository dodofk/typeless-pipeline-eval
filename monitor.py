#!/usr/bin/env python3
"""monitor.py —— 跑實驗時在旁邊採樣 RAM / CPU / 溫度。

指標本身(CER / runtime / RTF / judge)在 grade.py。這一支只回答一件事:
**這次跑出來的速度可不可信?** 16GB 的機器上,同一個模型的 tok/s 可以差 2.7 倍,
差別全部來自別的東西在搶記憶體頻寬。沒有這份紀錄就分不出「模型變快了」跟
「剛好那次機器比較閒」。

兩種用法:

    # A. 包住一個指令 —— 跑完自動停,順便給你那段期間的摘要
    uv run python monitor.py --out logs/polish-run.jsonl -- \
        uv run python polish.py --asr asr/breeze.jsonl --out polish/x.jsonl

    # B. 自己跑,Ctrl-C 停
    uv run python monitor.py --out logs/m.jsonl --interval 5

溫度:macOS 的 `powermetrics` **要 sudo**,沒有 sudo 就量不到攝氏溫度。
沒有 sudo 時退而求其次記 `pmset -g therm` 的降頻警告(那個免 sudo),
以及 CPU 使用率與記憶體壓力 —— 熱到降頻的話這兩個看得出來。
要真的溫度就:
    sudo uv run python monitor.py --temp --out logs/m.jsonl -- <指令>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import signal
import statistics as st
import subprocess
import sys
import time

PAGE = 16384          # Apple Silicon 的 page size,下面會用 vm_stat 的實際值覆蓋


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def mem() -> dict:
    """vm_stat —— 免 sudo。回傳 GB 與百分比。"""
    out = _run(["vm_stat"])
    if not out:
        return {}
    page = PAGE
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    g = {}
    for k, label in [("Pages free", "free"), ("Pages active", "active"),
                     ("Pages inactive", "inactive"), ("Pages wired down", "wired"),
                     ("Pages occupied by compressor", "compressed")]:
        mm = re.search(rf"{k}:\s+(\d+)", out)
        if mm:
            g[label] = int(mm.group(1)) * page / 2**30
    total = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0) / 2**30

    # ⚠️ vm_stat 的 "Pages free" 在 macOS 上幾乎永遠接近 0 —— 系統會把閒置記憶體
    #    全部拿去當快取,那不是「不夠用」。真正該看的是 memory_pressure 回報的
    #    可回收後剩餘比例,以及 swap 有沒有在動。
    free_pct = None
    mp = _run(["memory_pressure", "-Q"])
    mm = re.search(r"free percentage:\s*(\d+)", mp)
    if mm:
        free_pct = int(mm.group(1))

    swap_used = None
    sw = _run(["sysctl", "-n", "vm.swapusage"])
    mm = re.search(r"used\s*=\s*([\d.]+)M", sw)
    if mm:
        swap_used = round(float(mm.group(1)) / 1024, 2)

    return {"total_gb": round(total, 2) if total else None,
            "mem_free_pct": free_pct,
            "swap_used_gb": swap_used,
            "swapouts": _swapouts(out),
            "wired_gb": round(g.get("wired", 0), 2),
            "compressed_gb": round(g.get("compressed", 0), 2)}


def _swapouts(vm_stat_out: str) -> int | None:
    """累計 swapout 次數。兩筆之間有增加 = 這段期間真的在換頁,那才是記憶體不夠。"""
    m = re.search(r"Swapouts:\s+(\d+)", vm_stat_out)
    return int(m.group(1)) if m else None


def cpu_and_load() -> dict:
    """iostat 取一秒的 user/sys/idle 與 load average —— 免 sudo。

    不用 `top -l 1`:top 的第一筆是開機以來的累計值,不是當下。"""
    out = _run(["iostat", "-c", "2", "-w", "1"], timeout=20)
    rows = [l.split() for l in out.strip().splitlines() if l.strip()]
    nums = [r for r in rows if r and re.match(r"^[\d.]+$", r[0])]
    if not nums:
        return {}
    r = nums[-1]                       # 第二筆才是這一秒的
    try:
        return {"cpu_user": float(r[-6]), "cpu_sys": float(r[-5]), "cpu_idle": float(r[-4]),
                "load_1m": float(r[-3]), "load_5m": float(r[-2]), "load_15m": float(r[-1])}
    except (ValueError, IndexError):
        return {}


def thermal(want_temp: bool) -> dict:
    """降頻警告免 sudo;攝氏溫度要 sudo。"""
    d = {}
    t = _run(["pmset", "-g", "therm"])
    d["throttled"] = ("No thermal warning level has been recorded" not in t
                      and "thermal" in t.lower())
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", t)
    if m:
        d["cpu_speed_limit_pct"] = int(m.group(1))
    if want_temp:
        if os.geteuid() != 0:
            d["temp_c"] = None
            d["temp_note"] = "要 sudo 才量得到溫度"
        else:
            out = _run(["powermetrics", "-n", "1", "-i", "200", "--samplers", "thermal"], 30)
            mm = re.findall(r"([\d.]+)\s*C", out)
            d["temp_c"] = max(float(x) for x in mm) if mm else None
            if d["temp_c"] is None:
                d["temp_note"] = "powermetrics 沒回報溫度(機型可能不支援)"
    return d


def heavy_procs(n: int = 5) -> list[dict]:
    """吃 CPU 最兇的幾個 —— 用來事後回答「那次是誰在搶」。"""
    out = _run(["ps", "-Ao", "pcpu,rss,comm", "-r"])
    rows = []
    for line in out.splitlines()[1:n + 1]:
        p = line.split(None, 2)
        if len(p) == 3:
            try:
                rows.append({"cpu": float(p[0]), "rss_gb": round(int(p[1]) / 2**20, 2),
                             "cmd": p[2].strip()[:60]})
            except ValueError:
                pass
    return rows


def sample(want_temp: bool) -> dict:
    s = {"t": round(time.time(), 2)}
    s.update(mem())
    s.update(cpu_and_load())
    s.update(thermal(want_temp))
    s["top"] = heavy_procs()
    return s


def summarize(samples: list[dict]) -> dict:
    def col(k):
        return [s[k] for s in samples if isinstance(s.get(k), (int, float))]

    def stats(k):
        v = col(k)
        if not v:
            return None
        return {"min": round(min(v), 2), "mean": round(st.mean(v), 2), "max": round(max(v), 2)}

    busiest = {}
    for s in samples:
        for p in s.get("top", []):
            busiest[p["cmd"]] = max(busiest.get(p["cmd"], 0), p["cpu"])
    return {
        "n_samples": len(samples),
        "duration_s": round(samples[-1]["t"] - samples[0]["t"], 1) if len(samples) > 1 else 0,
        "mem_free_pct": stats("mem_free_pct"),
        "swap_used_gb": stats("swap_used_gb"),
        "swapouts_delta": (samples[-1]["swapouts"] - samples[0]["swapouts"]
                           if samples[0].get("swapouts") is not None
                           and samples[-1].get("swapouts") is not None else None),
        "cpu_idle": stats("cpu_idle"),
        "load_1m": stats("load_1m"),
        "temp_c": stats("temp_c"),
        "throttled_any": any(s.get("throttled") for s in samples),
        "busiest": sorted(busiest.items(), key=lambda x: -x[1])[:5],
    }


def verdict(sm: dict) -> tuple[bool, list[str]]:
    """速度數字可不可信。標準寫死在這裡,不要每次憑感覺講。"""
    why = []
    if sm.get("load_1m") and sm["load_1m"]["mean"] > 4:
        why.append(f"load average 平均 {sm['load_1m']['mean']}(>4 表示有人在搶 CPU)")
    if sm.get("cpu_idle") and sm["cpu_idle"]["mean"] < 40:
        why.append(f"CPU idle 平均只有 {sm['cpu_idle']['mean']}%(<40% 表示機器很忙)")
    if sm.get("mem_free_pct") and sm["mem_free_pct"]["min"] < 15:
        why.append(f"可用記憶體最低到 {sm['mem_free_pct']['min']}%(<15% 會開始擠壓)")
    if sm.get("swapouts_delta"):
        why.append(f"期間 swap out {sm['swapouts_delta']} 次 —— 記憶體真的不夠,速度一定失真")
    if sm.get("throttled_any"):
        why.append("期間出現過降頻警告")
    others = [c for c, v in sm.get("busiest", []) if v > 50
              and not any(k in c for k in ("llama", "python", "whisper", "funasr"))]
    if others:
        why.append(f"有別的程式吃 CPU:{', '.join(others[:3])}")
    return (not why), why


def main() -> int:
    ap = argparse.ArgumentParser(description="採樣 RAM / CPU / 溫度",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--out", help="每一筆採樣寫成 JSONL")
    ap.add_argument("--interval", type=float, default=5.0, help="採樣間隔秒數(預設 5)")
    ap.add_argument("--temp", action="store_true", help="量攝氏溫度(要 sudo)")
    ap.add_argument("cmd", nargs="*", help="-- 後面接要監控的指令;不給就一直跑到 Ctrl-C")
    a = ap.parse_args()

    if a.temp and os.geteuid() != 0:
        print("⚠️ --temp 要 sudo 才量得到攝氏溫度;現在只會記降頻警告。", file=sys.stderr)

    out = pathlib.Path(a.out) if a.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")

    samples: list[dict] = []
    proc = subprocess.Popen(a.cmd) if a.cmd else None
    if proc:
        print(f"監控中:{' '.join(a.cmd)}\n", file=sys.stderr)

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sig)

    try:
        while not stop:
            s = sample(a.temp)
            samples.append(s)
            if out:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            if proc and proc.poll() is not None:
                break
            # cpu_and_load 自己花了約 1 秒,扣掉
            time.sleep(max(0.0, a.interval - 1.0))
    except KeyboardInterrupt:
        pass

    rc = proc.wait() if proc else 0
    if not samples:
        print("沒有採到樣本。", file=sys.stderr)
        return rc

    sm = summarize(samples)
    ok, why = verdict(sm)
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"採樣 {sm['n_samples']} 次 / {sm['duration_s']}s", file=sys.stderr)
    for k, label in [("mem_free_pct", "可用記憶體 %"), ("swap_used_gb", "swap 使用 GB"),
                     ("cpu_idle", "CPU idle %"), ("load_1m", "load 1m"), ("temp_c", "溫度 °C")]:
        v = sm.get(k)
        print(f"  {label:<16}" + (f"min {v['min']:>7}  mean {v['mean']:>7}  max {v['max']:>7}"
                                  if v else "—(量不到)"), file=sys.stderr)
    if sm.get("swapouts_delta") is not None:
        print(f"  {'期間 swap out 次數':<16}{sm['swapouts_delta']}", file=sys.stderr)
    if sm["busiest"]:
        print("  期間最吃 CPU 的:" +
              ", ".join(f"{c}({v:.0f}%)" for c, v in sm["busiest"][:4]), file=sys.stderr)
    print(f"\n  速度數字可信度:{'✅ 可信' if ok else '⚠️ 不可信'}", file=sys.stderr)
    for w in why:
        print(f"     - {w}", file=sys.stderr)
    if out:
        summary = out.with_suffix(".summary.json")
        summary.write_text(json.dumps({"summary": sm, "speed_trustworthy": ok, "why": why},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  → {out}  /  {summary}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
