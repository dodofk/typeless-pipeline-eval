"""計分編排 —— run 產生文字,scoring 算分數。兩者刻意分開。

分開的理由:改了 metric(調幻覺的 n、加一個新指標)之後,要能對**所有歷史 run**
重算,而不是把模型再跑一次。`./tl score <run_id>` 走的就是這條路。
"""

from __future__ import annotations

from statistics import mean

from .metrics import asr as m_asr
from .metrics import polish as m_polish


def _nums(items, path):
    """把每個 item 的某個巢狀欄位撈成數字 list,None 跳過。"""
    out = []
    for it in items:
        v = it.get("metrics") or {}
        for k in path:
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(k)
        if isinstance(v, (int, float)):
            out.append(v)
    return out


def _avg(items, path):
    v = _nums(items, path)
    return round(mean(v), 4) if v else None


def _ratio(items, in_path, out_path):
    """移除率 = 1 - 總殘留 / 總輸入。輸入本來就是 0 的話回 None,不是 1.0。"""
    a, b = _sum(items, in_path), _sum(items, out_path)
    return round(1 - b / a, 4) if a else None


def _sum(items, path):
    v = _nums(items, path)
    return sum(v) if v else 0


def score_items(run, dataset) -> None:
    """就地重算 run.items 每一筆的 metrics。run.items 必須已經有 id / raw / out。"""
    for it in run.items:
        item = dataset.get(it["id"])
        if item is None:
            it["metrics"] = {"error": f"dataset {dataset.name} 裡沒有 id={it['id']}"}
            continue
        # run 存的 raw 是權威(當時真的送進去的東西),dataset 只補 ref/terms/dur
        item.raw = it.get("raw", item.raw)

        if run.arm == "asr":
            it["metrics"] = m_asr.score(item, it.get("out", ""),
                                        (it.get("timings") or {}).get("wall_s"))
        else:
            it["metrics"] = m_polish.score(item, it.get("out", ""), it.get("timings"))
            if "judge" in it:                    # judge 是另外跑的,保留不覆蓋
                it["metrics"]["judge"] = it["judge"]


def aggregate(run) -> dict:
    """跨 item 總計。比率取平均,計數取總和 —— 混著用會誤導。"""
    items = [i for i in run.items if not (i.get("metrics") or {}).get("error")]
    if not items:
        return {"n": 0}

    if run.arm == "asr":
        tr = [(i["metrics"].get("term_recall") or {}) for i in items]
        hit = sum(t.get("hit", 0) for t in tr)
        tot = sum(t.get("total", 0) for t in tr)
        return {
            "n": len(items),
            "cer": _avg(items, ["cer"]),
            "term_recall": round(hit / tot, 4) if tot else None,
            "term_hit": f"{hit}/{tot}" if tot else None,
            "len_ratio": _avg(items, ["len_ratio"]),
            "rtf": _avg(items, ["rtf"]),
            "simp_chars": _sum(items, ["simp_chars"]),
        }

    tk = [(i["metrics"].get("term_keep") or {}) for i in items]
    hit = sum(t.get("hit", 0) for t in tk)
    tot = sum(t.get("total", 0) for t in tk)
    jud = [i["metrics"].get("judge") for i in items if i["metrics"].get("judge")]
    return {
        "n": len(items),
        # 清理:殘留是絕對數量(0 才叫乾淨),移除率是比例
        "filler_a_out": _sum(items, ["filler_a", "out"]),
        "filler_a_in": _sum(items, ["filler_a", "in"]),
        # 移除率用**總量**基準,不是每個 clip 的平均 —— 平均會被短 clip 拉歪。
        # 坑 #3 引用的「v1 移除 0 個、v2 移除 27%」就是總量口徑。
        "filler_a_removed": _ratio(items, ["filler_a", "in"], ["filler_a", "out"]),
        "filler_a_removed_mean": _avg(items, ["filler_a", "removed"]),
        "filler_b_out": _sum(items, ["filler_b", "out"]),
        "stutter_out": _sum(items, ["stutter", "out"]),
        "stutter_removed": _ratio(items, ["stutter", "in"], ["stutter", "out"]),
        "simp_out": _sum(items, ["simp", "out"]),
        # 護欄
        "len_ratio_raw": _avg(items, ["len_ratio_raw"]),
        "len_ratio_ref": _avg(items, ["len_ratio_ref"]),
        "cer_ref": _avg(items, ["cer_ref"]),
        "term_keep": round(hit / tot, 4) if tot else None,
        # 幻覺
        "halluc_rate": _avg(items, ["halluc", "rate"]),
        "halluc_chars": _sum(items, ["halluc", "novel_chars"]),
        "drift_n": sum(j.get("n_drift", 0) for j in jud) if jud else None,
        "drift_high": sum(j.get("n_high", 0) for j in jud) if jud else None,
        "judged": len(jud) or None,
        # 速度。可信度看 env.speed_trustworthy。
        "latency_s": _avg(items, ["latency_s"]),
        "tok_s": _avg(items, ["tok_s"]),
        "gen_tok": _sum(items, ["gen_tok"]),
    }


def rescore(run, dataset) -> None:
    score_items(run, dataset)
    run.aggregate = aggregate(run)
