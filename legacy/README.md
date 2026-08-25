# 舊的 `./tl` 那一套 —— 保留不刪,但已經不是主線

2026-08-25 收斂 metric 的時候搬過來的。**沒有刪掉**,因為裡面的數字是歷史紀錄,
而且 `runs/` 裡的舊 run 要靠這些模組才讀得懂。

搬走的東西跟為什麼:

| 檔案 | 原本做什麼 | 為什麼不留在主線 |
|---|---|---|
| `tl` / `cli.py` | 子指令 CLI(run/score/judge/report/show/list/legacy) | 太多層。主線改成三支獨立腳本 |
| `registry.py` | run 目錄 + index.jsonl | 主線改成一個 JSONL 就是一次結果 |
| `report.py` | 跨 run 對照表 | 指標剩三個,不需要專門的排版模組 |
| `scoring.py` | 計分編排 + aggregate | 併進 `grade.py` |
| `dataset.py` | evalset 載入 | 併進 `grade.py` 的 `load_ref()` |
| `asr_sources.py` | 從 OpenWhispr SQLite 撈結果 | 主線不自動抓任何東西,只吃檔案 |
| `lexicon.py` | 語助詞 / 口吃的詞表與計數 | 指標被砍掉了(見下) |
| `metrics/` | asr / polish / halluc 三個模組 | 同上 |
| `tests/` | norm 與 halluc 的測試 | halluc 那組沒有被測對象了 |

## 被砍掉的指標,以及砍掉的理由

| 指標 | 為什麼砍 |
|---|---|
| tier-A / tier-B 語助詞殘留與移除率 | 詞表要維護,而且「啊」這種要看句位的字算法很難講清楚 —— 交接成本高過它的價值 |
| 口吃殘留與移除率 | 同上,而且疊字的正則會誤傷「看看」「謝謝」這種合法疊字 |
| 簡體殘留字數 | 有用但太細;需要的話 `norm.simplified_chars()` 還在 |
| 長度比(vs 輸入 / vs 正解) | 判斷「刪過頭」現在交給 judge 的 quality 分數 |
| n-gram 幻覺率 | 判斷「掰東西」現在完全交給 judge |
| 術語召回 / 保留 | `manifest.terms` 一直是空的,從來沒真的量到過 |

要拿回來的話,這裡的檔案都還能跑,把 `legacy/` 加進 `sys.path` 就好。

## 2026-08-25 第二批:舊腳本與舊資料

`legacy/scripts/` —— 2026-08-18 TTS bakeoff 那一輪用的腳本。**沒有刪**,
`out/RESULTS_20260818_tts_bakeoff.md` 的數字是它們跑出來的,要回頭查得靠它們。

| 檔案 | 做什麼 | 現在用什麼 |
|---|---|---|
| `gt.py` | 打各家 ASR API 產 ground truth | `asr.py` |
| `run_eval.py` | 跑本機 ASR + 比對 | `asr.py` + `grade.py` |
| `score.py` | 算 CER / 術語召回 | `grade.py` |
| `bench_polish.py` / `bench_polish_en.py` | 潤稿 A/B | `polish.py` + `grade.py` |
| `bonsai_polish.py` | Bonsai-27B 專用潤稿 | `polish.py --url` 指過去就好 |
| `probe_lang.py` | 探測語言參數的影響 | 一次性,沒有取代品 |
| `export_corpus.py` | 產 TTS 測試語料 | evalset 取代 |
| `e2e_real.py` | 端到端煙霧測試 | `asr.py` + `polish.py` |

`legacy/data/` —— 舊的測試資料,已被 `evalset/` 取代:

| 目錄 | 是什麼 |
|---|---|
| `corpus/` + `corpus.jsonl` | TTS 合成的 5 段語料。合成語音沒有真實的口吃與環境噪音,測不出真實表現 —— 這是後來改用真人錄音的原因 |
| `gold/` | 上面那批的人工正解 |
| `audio/` | 更早的煙霧測試音檔(s_long / s_short / smoke) |

`.ornith_expect` / `.qwen_expect`(各 11 bytes,只有一個檔案大小數字,
沒有任何東西引用)已刪除。
