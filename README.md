# local-typeless — 本機語音轉文字管線的評測工具

量的是這條管線:**口說 → ASR → 逐字稿 → LLM 潤稿 → 可以直接用的文字**。
全部跑在本機(M1 Pro 16GB),只有 judge 和雲端 ASR 會連網。

目標是讓「換一個模型 / 換一版 prompt,然後知道變好還是變壞」變成一行指令,
而不是每次重寫一支腳本。

---

## 30 秒版本

```bash
# 起服務(這些腳本不負責啟動模型)
./run_qwen35.sh &                     # 潤稿 LLM, :8902
./run_breeze_server.sh &              # Breeze-ASR-25, :8380

# 跑
uv run python asr.py    --engine breeze --out asr/breeze.jsonl
uv run python polish.py --asr asr/breeze.jsonl --out polish/mine.jsonl
uv run python grade.py  --asr asr/breeze.jsonl --polish polish/mine.jsonl \
                        --ref evalset/ --judge --per-item
```

輸出長這樣:

```
=== ASR ===
  n              14   (有正解可算 CER 的:11)
  CER            平均 11.18%   中位 8.89%
  RTF            平均 0.121   中位 0.111        (處理秒數/音檔秒數)
  runtime        總計 177.6s   平均 12.7s   最慢 29.7s

=== POLISH ===
  RTF            平均 0.100   中位 0.099
  runtime        總計 160.0s   平均 11.4s   最慢 31.6s
  品質(1-5)     平均 4.14   最低 2        已判 14 題
  幻覺           4 題有   其中 high 4 處
```

---

## 環境

需要 [`uv`](https://docs.astral.sh/uv/)。Python 3.13,相依只有 `zhconv`。

```bash
git clone <repo> && cd local-typeless
uv sync
```

模型檔和引擎的 build 都**不在 repo 裡**(太大),要自己準備:

| 放哪 | 是什麼 | 哪裡拿 |
|---|---|---|
| `models/qwen3.5-4b/` | 潤稿 LLM 的 GGUF | HuggingFace |
| `models/breeze-asr/` | Breeze-ASR-25 | HuggingFace |
| `models/funasr-gguf/` | Fun-ASR encoder + Qwen3 decoder + fsmn-vad | HuggingFace |
| `models/ggml-large-v3-turbo-q5_0.bin` | whisper.cpp | HuggingFace |
| `models/whisper-vad/ggml-silero-v5.1.2.bin` | Silero VAD | `ggml-org/whisper-vad` |
| `fun-asr/`, `transcribe.cpp/` | 引擎的 build | 各自的 repo |

`whisper-cli` 用 `brew install whisper-cpp`。

雲端的部分(選用):

```bash
export OPENAI_API_KEY=...        # judge 用
export ELEVENLABS_API_KEY=...    # scribe ASR 用
```

---

## 四支腳本

沒有子指令、沒有 CLI 框架。每一支 `--help` 都有完整說明。

### `asr.py` — 跑 ASR

```bash
uv run python asr.py --engine breeze --out asr/breeze.jsonl
```

引擎:`whisper` / `whisper:no-vad` / `funasr:q4km` / `breeze` / `scribe`

- 逐筆寫檔,中途掛掉不會全白跑
- 再跑一次自動跳過已完成的 id(`--overwrite` 可覆蓋)
- `--only id1,id2` 只跑幾筆

⚠️ **whisper 走這裡跟走 OpenWhispr app 不保證等價** —— app 的上傳路徑會開本機 VAD,
而且那組參數跟 `whisper-cli` 自己的預設有四個不同(已在 `typeless/asr_engines.py`
複製了 app 的那組)。funasr / breeze / scribe 則是等價的,因為 app 只是把整個檔
POST 出去。

### `polish.py` — 跑潤稿 LLM

```bash
uv run python polish.py --asr asr/breeze.jsonl --out polish/mine.jsonl
```

- 輸入就是 `asr.py` 的輸出
- `--temp 0` 是預設。**A/B 比較一定要 temp=0**,不然差異可能純粹是雜訊
- prompt 用 `--prompt <檔案>` 換,檔案的 sha8 會記進輸出 —— prompt 改了但檔名沒改的話,只有 sha 認得出來

### `grade.py` — 算分

```bash
uv run python grade.py --asr asr/breeze.jsonl --polish polish/mine.jsonl \
                       --ref evalset/ --judge --per-item --out result.json
```

`--asr` 跟 `--polish` 至少要給一個,兩個都給就兩層一起評。
**預設完全不連網**,只有 `--judge` 會。

### `monitor.py` — 跑的時候採樣資源

```bash
uv run python monitor.py --out logs/run.jsonl -- \
    uv run python polish.py --asr asr/breeze.jsonl --out polish/mine.jsonl
```

同一個模型在這台機器上量過 27.5 tok/s 和 10.1 tok/s,差 2.7 倍,
差別全部來自別的程式在搶記憶體頻寬。**要比速度就一定要有這份紀錄。**

跑完會給 `speed_trustworthy` 判定與原因。攝氏溫度要 `sudo`(powermetrics 的限制)。

---

## 指標

| 指標 | 怎麼算 | 需要 |
|---|---|---|
| **CER** | 正規化後的編輯距離 ÷ 正解長度 | reference |
| **runtime_s** | 每題牆鐘秒數 | — |
| **RTF** | `runtime_s ÷ 音檔秒數`,越小越快 | 音檔長度 |
| **judge:幻覺** | LLM 判定,含 span 與嚴重度 | `--judge` |
| **judge:品質** | LLM 給 1–5 分 | `--judge` |

兩層用同一組。**算不出來的一律印 `—`,不會印成 0** —— 「沒量到」跟「量到 0」
是完全不同的兩件事。

CER 的正規化(`typeless/norm.py`,全專案只有這一份):繁簡統一、去標點、
NFKC、**中文數字轉阿拉伯數字**。最後一項以前沒做,害歷史 CER 全部高估約 6 個百分點。

judge 的完整判準就是 `typeless/judge.py` 裡的 `JUDGE_PROMPT`,可以直接讀。
它會先分類「這是清理稿還是重寫」—— 如果逐字稿裡的人說「幫我轉成精簡的 plan」
而模型照做了,那是重寫,品質最高 2 分。那是最重要的一種失敗。

judge 是 **OpenAI 相容**的,不綁供應商:

```bash
JUDGE_BASE_URL   預設 https://api.openai.com/v1
JUDGE_MODEL      預設 gpt-5.4-mini
JUDGE_API_KEY    沒設就退回 OPENAI_API_KEY
JUDGE_URL        完整 endpoint 覆寫(路徑不標準的供應商用)
```

vLLM、Ollama、groq、together、本機 llama-server 都可以。

細節看 [`METRICS.md`](./METRICS.md)。

---

## 接自己的 pipeline

`grade.py` 只讀檔案,不自動抓任何東西。你的 pipeline 只要把結果轉成
JSONL 就能評 —— 轉檔在你那邊做:

```jsonc
// --asr      ASR 逐字稿,同時也是潤稿層的 input
{"id":"teach-01","text":"...","dur_s":106.8,"elapsed_s":12.4}

// --polish   潤稿輸出
{"id":"teach-01","text":"...","input":"...","latency_s":15.0}

// --ref      正解,欄位全部選填
{"id":"teach-01","asr_ref":"...","polish_ref":"...","dur_s":106.8}
```

只有 `id` 和 `text` 是必填。三個路徑都可以改成「一個資料夾裝 `<id>.txt`」。

完整格式看 [`FORMATS.md`](./FORMATS.md)。

---

## 資料集

```
evalset/
  manifest.jsonl          一行一 item,權威索引
  audio/<id>.wav          真人錄音(不進版控)
  text/<id>.zh-cn.txt     ASR 原始輸出,一字未動
  text/<id>.zh-tw.txt     上面那個轉繁 + 人工修正 = 逐字稿
  _discarded/             剔除的 item + 原因
```

14 個 clip,27 分鐘,真人錄的中英夾雜。三種場景:`teach` 教學、
`agent` 對 AI 口述指令、`meeting` 會議。

⚠️ **`.zh-tw.txt` 是逐字稿,不是潤稿正解。** 它保留語助詞,拿它當潤稿的答案會讓
指標整個反過來 —— 什麼都不刪的模型 CER 最低。它掛在 `asr_ref`。
潤稿層的正解(`manifest.gold_final`)目前全部是 null,所以**潤稿層的 CER 還量不到**,
品質只能靠 judge。

補標註**不用重跑模型**:hypothesis 在 `asr/`、`polish/`,reference 在 `evalset/`,
靠 id 對接。改完 reference 重跑一次 `grade.py`,幾秒。

---

## 目錄

```
asr.py polish.py grade.py monitor.py     主線,就這四支
typeless/
  norm.py         正規化 + CER(唯一一份)
  judge.py        LLM judge(OpenAI 相容)
  engines.py      llama-server client
  asr_engines.py  ASR 引擎驅動
funasr_server.py breeze_server.py        本機 ASR 的 OpenAI 相容 shim
run_*.sh                                 起 llama-server 的腳本
evalset/                                 資料集
prompts/cleanup-zhTW-mixed-v2.txt        潤稿 prompt(目前這一版)
```

## 已知的坑

跑之前值得先讀,都是實際踩過的:

1. **正規化只能有一份。** 曾經分裂成 `gt.py` 和 `score.py` 兩份且不一致,
   同一組輸出兩邊算出來的 CER 差 8–28%,全部是測量誤差。現在只有 `typeless/norm.py`。
2. **A/B 一定要 `temp=0`。** temp>0 的兩次跑,差異可能純粹是取樣雜訊。
3. **速度只有獨佔機器時可比。** 用 `monitor.py`,不要憑感覺。
4. **`--engine whisper` 預設開 VAD。** 那是為了對齊 OpenWhispr 的上傳路徑,
   不是 `whisper-cli` 的預設。安靜的錄音(-51 dB)開 VAD 會被吃掉九成內容。
5. **「沒量到」不要寫成 0。** 缺 reference、judge 失敗,都要印 `—` 並另外報。

## 目前的結論(2026-08-25)

ASR 用 **breeze**:CER 10.46%(人工修過的 ref),是唯一直接輸出繁體的。
比 funasr 慢 65% 但 RTF 只有 0.121,遠低於即時。

端到端 RTF **0.221** —— 5 分鐘的錄音約 66 秒處理完,ASR 佔 53%、潤稿佔 47%。

完整數字看 [`RESULTS_20260825_ASR.md`](./RESULTS_20260825_ASR.md)。
