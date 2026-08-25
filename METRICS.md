# metric codebase & pipeline

`口說 → STT → 逐字稿 → LLM 潤稿 → 結構化文字`，兩層各自量。

一行指令換一個模型 / 一版 prompt，然後知道它變好還變壞：

```bash
./tl run --dataset evalset --label qwen35-v2 \
         --url http://127.0.0.1:8902 --model qwen3.5-4b \
         --prompt prompts/cleanup-zhTW-mixed-v2.txt --temp 0 --seed 1234
./tl report --arm polish
```

## 這套東西不做什麼

**不啟動模型。** llama-server 由你自己起（`./run_qwen35.sh` 等）。
**不跑 ASR。** 逐字稿是 OpenWhispr（或其他 code）的副產物，這裡只讀檔評分。

## 模組

| 檔案 | 負責 |
|---|---|
| `typeless/norm.py` | **唯一的正規化來源**。繁簡、標點、大小寫、中文數字。所有 metric 都走這裡 |
| `typeless/lexicon.py` | 語助詞 tier A/B、口吃的定義。中英用不同正規化視角 |
| `typeless/dataset.py` | evalset 格式定義與載入 |
| `typeless/engines.py` | llama-server adapter + 環境快照（獨佔偵測） |
| `typeless/judge.py` | 語意漂移的 LLM-as-a-judge（MiniMax-M3，雲端） |
| `typeless/metrics/asr.py` | CER、術語召回、長度比、RTF、字體 |
| `typeless/metrics/polish.py` | 語助詞/口吃/簡體殘留、術語保留、長度比、幻覺、latency |
| `typeless/metrics/halluc.py` | **幻覺率**（新）—— 潤稿層的 CER |
| `typeless/registry.py` | run record 的讀寫 |
| `typeless/scoring.py` | 計分編排。跟「跑模型」刻意分開 |
| `typeless/report.py` | 跨 run 對照表 |

## CLI

```
./tl run      跑潤稿 + 計分 + 存成一筆 run
./tl score    對既有 run 重算分數（改了 metric 不用重跑模型）
./tl judge    對既有 run 補跑語意漂移判定
./tl report   跨 run 對照表
./tl show     看單一 clip 的輸入輸出與被標記的片段
./tl list     列出所有 run
./tl legacy   把 out/ 的歷史輸出匯成 run
./tl reindex  重建 runs/index.jsonl
```

## 外部 pipeline 怎麼接 —— `grade.py`

`./tl` 是「我自己跑」的路徑：它會叫模型、會去 OpenWhispr 的 SQLite 撈結果。

如果潤稿是你自己的 pipeline 跑的，用 **`grade.py`**。它只讀檔案，不連任何東西：

```
uv run python grade.py --asr asr/whisper.jsonl \
                       --polish polish/mine.jsonl \
                       --ref evalset/ --per-item --out result.json
```

metric 的實作是同一份（`typeless/metrics/`），所以兩條路的數字一定一致 ——
已驗過 `grade.py` 對 `runs/…-spokenly-baseline-qwen35-v2` 重算出的 aggregate
跟 `./tl` 存的完全相同。

交換格式（JSONL，三個檔用 `id` 對接）看 [`FORMATS.md`](./FORMATS.md)。
格式轉換是呼叫端的事，這支不猜也不自動掃描。

## evalset 格式

支援兩種 layout，靠有沒有 `manifest.jsonl` 自動判斷。

### A. Spokenly layout（`evalset/` 現在用的）

```
evalset/
  manifest.jsonl        一行一 item，權威索引
  audio/<id>.wav        原始錄音（這套不碰）
  raw/<id>.json         Spokenly 的 dictation 匯出（含 ASR 原文與各 stage）
  text/<id>.zh-cn.txt   ASR 原始輸出，一字未動
  text/<id>.zh-tw.txt   上面那個轉繁 + 人工修正
```

欄位怎麼對到 metric：

| Item 欄位 | 來源 | 用途 |
|---|---|---|
| `raw` | `raw/<id>.json` 的 `stage kind='original'` | 潤稿層的 input |
| `asr_ref` | `text/<id>.zh-tw.txt` | **ASR 層**的 reference（逐字稿） |
| `ref` | manifest 的 `gold_final` | **潤稿層**的 reference（目前 null） |
| `dur_s` | manifest 的 `duration_sec` | 算 RTF |
| `terms` | manifest 的 `terms` | 術語召回 / 保留（目前全空） |

**`.zh-tw.txt` 不能當潤稿 reference。** 它保留語助詞（全套 47 個 tier-A），
拿它算 CER 的話「什麼都不刪」分數最高、清得最乾淨分數最低 —— 實測 `meeting-05`
一個語助詞都沒刪拿到 0.7%（最好），`agent-03` 積極清理拿到 47.4%（最差）。
要量潤稿層的 CER 必須先有 `gold_final`。

`transcript_is_parent_full=true` 的 item（`teach-03a/b/c`）共用母檔的完整逐字稿，
不是自己那一段的 —— 預設跳過並印出原因（`--include-unusable` 可覆蓋）。

### B. 簡單 layout（沒有 manifest.jsonl 時）

```
evalset/
  text/<id>-raw.txt     ASR 逐字稿      = 潤稿層的 input
  text/<id>-tw.txt      人工潤過的正解  = 潤稿層的 reference
  text/<id>-asr.txt     （選用）人工逐字稿 = ASR 層的 reference
  meta.jsonl            （選用）補 dur_s / terms / tags
```

兩種 layout 都一樣：只有 input 也跑得動 —— 無參考那組 metric（語助詞殘留、
口吃、簡體、長度比、幻覺、judge）永遠算得出來。reference 一補上，CER /
術語保留自動生效。量不了的顯示 `—`，不是 0。

## ASR 從哪來 —— 兩端不能弄反

```
ground truth ← Spokenly + ElevenLabs + 人工修正 → evalset/text/<id>.zh-tw.txt
hypothesis   ← OpenWhispr audio upload，抽換不同 ASR 模型 → --asr-source
```

**raw 屬於 run，不屬於 dataset。** 同一批音檔換一個 ASR 模型就是另一份
hypothesis，全部對同一份 ground truth 計分。所以 `evalset` 的 `Item.raw`
預設是空的，沒給 `--asr-source` 會直接擋下來 —— 不給的話最容易犯的錯就是
拿 Spokenly 的輸出當 input，那等於用答案當考卷。

| `--asr-source` | 是什麼 |
|---|---|
| `openwhispr:<folder>` | OpenWhispr audio upload 的結果，限定某個 folder |
| `openwhispr` | 全部的 upload note |
| `openwhispr-polished` | OpenWhispr 自己的 AI 潤稿（拿來當對照組） |
| `spokenly[:stage]` | Spokenly/ElevenLabs 原文。⚠️ 是 GT 的來源引擎，分數是上界不是實力 |
| `dir:<path>` | 一個目錄的 `<id>.txt`，任何 ASR 的逃生出口 |

跑完會印對帳：對上幾個、dataset 有但 ASR 沒有的、ASR 有但 dataset 沒有的、
同一個檔傳了兩次的。靜靜少跑幾個 item 比報錯更糟。

### ⚠️ OpenWhispr 的 notes 表不記 ASR 模型

audio upload 走 `saveUploadNote()` → `notes` 表，欄位有 `source_file`（原始檔名，
接合鍵）、`content`（ASR 逐字稿）、`enhanced_content`（OpenWhispr 的 AI 潤稿）、
`folder_id`、`audio_duration_seconds` —— 但**沒有 `provider` / `model`**。

同一批音檔換三個 ASR 模型各上傳一次，三批 note 在資料庫裡分不出來。

**做法：一個 ASR 模型建一個 folder，各自上傳進去**，然後
`--asr-source openwhispr:<folder>`。零程式碼修改。
退路是 `--since` / `--until` 靠 `created_at` 切批次。

## 速度數字什麼時候可信

`env` 每次都記 `llama_procs`、`loadavg`、`mem_free_pct`、`heavy_procs`。
這台是記憶體頻寬綁定的 16GB 機器，實測同一個模型同一份 prompt：
乾淨時 **27 tok/s**，WindowServer / 瀏覽器 / 另一個 agent 在跑時 **10 tok/s**。

所以坑 #7 不只是「不要同時起兩個 llama-server」——任何吃頻寬的行程都算。
`speed_trustworthy=false` 時報表會標 `⚠速度` 並列出當時的重負載行程。

## run record

```
runs/<run_id>/run.json         config + 每個 item 的原文與分數 + 總計
runs/<run_id>/items/<id>.txt   模型實際輸出的文字
runs/index.jsonl               一行一 run，report 只讀這個
```

`run.json` 一定存 `raw` 和 `out` 的原文 —— 這是「改了 metric 可以對歷史 run
重算」的前提。舊腳本算完就印、數字不落地，換一版 metric 就得把全部模型再跑一次。

`config.polish` 記 `model / prompt_file / prompt_sha / temp / seed / top_p / top_k`。
`prompt_sha` 存在的理由：prompt 檔改了但檔名沒改的話，只有 sha 認得出來。

## 幻覺 metric 怎麼運作

分兩層，互補：

**L1 `halluc.py` —— 字面新內容。** 決定性、零成本、每次都跑。
輸出的字元 n-gram（中文 3、英文 4）在輸入裡找不到 → 標記。

三道防偽陽性：
1. **canon 先過** —— 繁簡、標點、大小寫、中文數字、術語斷字都不算幻覺
2. **來源變體** —— 輸入的原文、收合口吃後、再去語助詞後，任一命中就算有依據。
   因為那些刪除是 prompt 要求的合法行為，接縫不該算幻覺
3. **有界子序列** —— 去冗會刪掉一般字（`檢查的項目`→`檢查項目`），
   視窗允許中間跳過 ≤2 個字元。**有界**很重要：無界的話任何字串都能湊出來

定義因此是：**連「照 prompt 該刪的都刪掉」之後也推導不出來的內容。**

**L2 `judge.py` —— 語意漂移。** L1 抓不到「字都在輸入裡但被重組成新主張」。
MiniMax-M3，temp=0，輸出結構化 `{output_span, input_basis, kind, severity, why}`。
`--judge` 才跑（要 `MINIMAX_API_KEY`，慢，是 reasoning model）。

兩層真的互補 —— 實測 clip 18：L1 抓到 `TypeScript`（輸入只有 `T S mini`），
L2 沒標（它視為合法 ASR 修復）；L2 抓到子句融接（判 `merged`/`high`），
L1 只標到接縫的兩個字。

`rate` 是排序訊號，真正要看的是 `spans` 清單和 judge 的 `why`。
`./tl show <run_id> <clip>` 會把兩層一起印出來。

`./tl score <run_id> --n-cjk 2` 可以改參數對歷史 run 重算,不用重跑模型 ——
用的 n 會寫進 `run.config.metrics`,所以分數永遠找得回它的計算條件。

## 切開兩層誤差(gold-input arm)

```bash
./tl run --dataset evalset --input asr  --label m-asr   ...   # 餵 ASR 逐字稿
./tl run --dataset evalset --input gold --label m-gold  ...   # 餵人工逐字稿
```

同一個潤稿模型跑兩次,差值就是「ASR 的錯誤讓潤稿層額外壞了多少」。
`--input gold` 需要 `text/<id>-asr.txt`。

## 報表上的警告標記

| 標記 | 意思 |
|---|---|
| `0.7?` | temp>0，這組 A/B 的勝負可能純粹是雜訊 |
| `⚠速度` | 跑的時候不只一個 llama-server，tok/s 不可跨 run 比 |
| `—` | 這個 metric 現在量不了（跟 0 是不同的意思） |

## 測試

```bash
uv run python -m tests.test_norm      # 正規化：等價寫法、繁簡中性、坑#4
uv run python -m tests.test_halluc    # 幻覺：3 個真陽性 + 7 個偽陽性防線
```

改了 `norm.py` 或 `halluc.py` 的參數之後，這兩支必須還是全過。
