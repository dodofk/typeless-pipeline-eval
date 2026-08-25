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

## evalset 格式

```
evalset/
  audio/<id>.wav        原始錄音。這套不碰，留給重跑 ASR 用
  text/<id>-raw.txt     ASR 逐字稿      = 潤稿層的 input
  text/<id>-tw.txt      人工潤過的正解  = 潤稿層的 reference
  text/<id>-asr.txt     （選用）人工逐字稿 = ASR 層的 reference
  meta.jsonl            （選用）補 dur_s / terms / tags
```

`meta.jsonl` 一行一 item：

```json
{"id":"18","dur_s":76.4,"terms":[["skill"],["typescript","ts"]],"tags":["real","meeting"]}
```

只有 `-raw.txt` 也跑得動 —— 無參考的那組 metric（語助詞殘留、口吃、簡體、
長度比、幻覺）永遠算得出來。`-tw.txt` 一補上，CER / 術語保留就自動生效。
`-asr.txt` 補上，ASR 層的 CER 才量得了（目前量不了，會顯示 `—` 不是 0）。

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
