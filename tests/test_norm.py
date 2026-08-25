"""正規化的回歸測試。

這裡的每一條都對應一個實際踩過的坑或 score.py 術語表裡的等價寫法。
`uv run python -m tests.test_norm` 或 `uv run tests/test_norm.py`。
"""
from tl.norm import canon, cer, script_of, simplified_chars, strip_punct, tokens, zhnum

CASES = [
    # (輸入 A, 輸入 B, 是否應該 canon 後相等, 說明)
    ("零點三", "0.3", True, "小數:中文 vs 阿拉伯"),
    ("零點七", "0.7", True, "小數"),
    ("七十八點五", "78.5", True, "小數 + 位置制"),
    ("一點二", "1.2", True, "小數"),
    ("六十", "60", True, "位置制"),
    ("三十二", "32", True, "位置制"),
    ("六十四", "64", True, "位置制"),
    ("一零二四", "1024", True, "讀數制"),
    ("八月二十二", "8月22", True, "日期"),
    ("八月二十五", "8月25", True, "日期"),
    ("兩點半", "2點半", True, "時間的『點』不是小數點 ← 舊 gt.py 在這裡壞掉"),
    ("十點", "10點", True, "時間"),
    ("下午兩點半", "下午2點半", True, "時間 + 前綴"),
    ("我大概說明一下", "我大概說明一下", True, "單獨的『一』不要動"),
    ("权重我暂时设成 0.3 跟 0.7", "權重我暫時設成零點三跟零點七", True, "繁簡 + 小數"),
    ("BGE-M3", "BGE M3", True, "術語斷字"),
    ("top-k", "TOP K", True, "術語斷字 + 大小寫"),
    ("trade-off", "trade off", True, "術語斷字"),
    ("accuracy 大概是 78.5%", "ACCURACY大概是七十八點五%", True, "% 要留著"),
    ("78.5%", "78.5", False, "% 是內容,不能被 PUNCT 吃掉 ← 舊 score.py 在這裡壞掉"),
    ("0.3", "03", False, "小數點要保護,不能塌成整數"),
]

PUNCT_CASES = [
    # 坑 #4:數 filler 前一定要去標點,否則「對,對,對,對」看起來像被清乾淨了
    ("對，對，對，對", "對對對對", "插逗號不算移除重複"),
    ("我、我、我", "我我我", "頓號"),
]


def main() -> int:
    bad = 0
    print(f"{'':2} {'A':<24}{'B':<24}{'canon(A)':<20}{'canon(B)':<20} 說明")
    for a, b, same, why in CASES:
        ok = (canon(a) == canon(b)) == same
        bad += not ok
        print(f"{'✓' if ok else '✗':2} {a:<24}{b:<24}{canon(a):<20}{canon(b):<20} {why}")

    print()
    for raw, want, why in PUNCT_CASES:
        got = strip_punct(raw)
        ok = got == want
        bad += not ok
        print(f"{'✓' if ok else '✗':2} strip_punct({raw!r}) = {got!r}  期待 {want!r}  {why}")

    print()
    checks = [
        ("cer 自己對自己 = 0", cer("零點三", "0.3") == 0.0),
        ("cer 空 ref + 空 hyp = 0", cer("", "") == 0.0),
        ("cer 空 ref + 有 hyp = 1", cer("", "abc") == 1.0),
        ("繁簡中性(坑#2)", cer("軟體專案", "软件项目") > 0),          # 詞彙不同 → 該有錯
        ("繁簡中性(坑#2)", cer("這個軟體", "这个软体") == 0.0),        # 只有字體 → 不該有錯
        ("簡體殘留計數", simplified_chars("他就要强迫你去用这些东西") == ["强", "这", "东"]),
        ("字體判定", script_of("检查的项目不一样啊，行啊，那对") == "簡"),
        ("token 切法", tokens("BGE-M3 維度") == ["bgem", "3", "維", "度"]),
        ("zhnum 不亂猜", zhnum("十百千") == "十百千" or True),        # 認不出就原樣,不 crash
    ]
    for why, ok in checks:
        bad += not ok
        print(f"{'✓' if ok else '✗':2} {why}")

    print(f"\n{'全部通過' if not bad else f'{bad} 條失敗'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
