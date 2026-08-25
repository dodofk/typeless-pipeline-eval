"""幻覺偵測的校準測試。

正例來自 brief §2 坑 #6 實際踩到的 case,負例來自「合法的潤稿行為不該被標記」。
改了 n / SLACK / 來源變體之後,這支必須還是全過。
"""
import pathlib

from typeless.metrics.halluc import novel_spans

ROOT = pathlib.Path(__file__).resolve().parent.parent


def spans_of(raw, out):
    return novel_spans(raw, out).spans


POSITIVE = [
    # (輸入, 輸出, 應該被標記的關鍵字, 說明)
    ("你沒幫我分享那一篇嗎？T S mini 上有",
     "你沒幫我分享那一篇嗎？TypeScript mini 上有",
     "typescrip", "坑#6 的原始 case:T S mini → TypeScript"),
    ("對有其實世界上更。這邊是有做壓縮過一次的嘛",
     "其實世界上有更壓縮過一次的",
     "有更", "兩個分開的子句被融接成一個新主張"),
    ("這邊是有做壓縮過一次的嘛",
     "這邊有做壓縮，我記得 Amy 也幫忙分享過",
     "amy", "憑空生出人名"),
]

NEGATIVE = [
    # (輸入, 輸出, 說明) —— 這些是 prompt 要求的合法行為,不該有任何標記
    ("呃，就是好，嗯，好，下一个是 TEST", "好，下一個是 test", "刪 tier-A 語助詞 + 繁簡 + 大小寫"),
    ("對對對對，或者你呃對對對對", "對，或者你，對", "收合口吃"),
    ("它大概減減少大概兩成", "大概減少兩成", "收合口吃造成的接縫"),
    ("检查的项目不一样啊", "檢查項目不一樣", "刪冗字造成的接縫"),
    ("权重我暂时设成零点三跟零点七", "權重我暫時設成 0.3 跟 0.7", "繁簡 + 中文數字轉阿拉伯"),
    ("BGE M3 的 TOP K 等於五", "BGE-M3 的 top-K 等於 5", "術語斷字修正"),
    ("先写 test 再写 code 吗", "先寫 test 再寫 code 嗎", "純繁簡"),
]


def main() -> int:
    bad = 0
    print("--- 應該抓到(真陽性) ---")
    for raw, out, want, why in POSITIVE:
        got = spans_of(raw, out)
        ok = any(want in s for s in got)
        bad += not ok
        print(f"{'✓' if ok else '✗':2} 期待含 {want!r:<14} 實得 {got}   {why}")

    print("\n--- 不該抓到(偽陽性防線) ---")
    for raw, out, why in NEGATIVE:
        got = spans_of(raw, out)
        ok = not got
        bad += not ok
        print(f"{'✓' if ok else '✗':2} 期待 []{'':<12} 實得 {got}   {why}")

    print("\n--- 真實資料上的排序(不是斷言,是拿來看的) ---")
    for clip in ("14", "16", "17", "18"):
        p = ROOT / "out" / f"e2e_{clip}_raw.txt"
        if not p.exists():
            continue
        raw = p.read_text()
        line = []
        for tag in ("qwen_v2t0", "ornith_v2", "bonsai"):
            q = ROOT / "out" / f"e2e_{clip}_polished_{tag}.txt"
            if q.exists():
                line.append(f"{tag}={novel_spans(raw, q.read_text()).rate * 100:.2f}%")
        print(f"   clip {clip}: " + "  ".join(line))

    print(f"\n{'全部通過' if not bad else f'{bad} 條失敗'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
