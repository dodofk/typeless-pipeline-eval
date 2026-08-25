# 評測 codebase —— 交接說明

量的東西:**口說 → ASR → 逐字稿 → LLM 潤稿 → 結構化文字**。
兩層分開評,因為它們壞的方式不一樣:ASR 錯在聽錯字,潤稿錯在改變意思。

## 四支腳本,沒有別的

```
asr.py       跑 ASR                → asr/<name>.jsonl
polish.py    跑潤稿 LLM            → polish/<name>.jsonl
grade.py     算分                  → 印表 / --out JSON
monitor.py   跑的時候在旁邊採樣資源   → logs/<name>.jsonl
```

全部用 `uv run python xxx.py`。沒有子指令、沒有 run registry、沒有 CLI 框架。
每一支都 `--help` 有完整說明。

模型**不由這些腳本啟動**。llama-server / funasr / breeze 要自己先跑起來。

## 指標只有這些

| 指標 | 怎麼算 | 需要什麼 |
|---|---|---|
| **CER** | 正規化後的編輯距離 ÷ 正解長度 | 要有 reference |
| **runtime_s** | 這一題花的牆鐘秒數 | 無 |
| **RTF** | `runtime_s ÷ 音檔秒數`,越小越快 | 要有音檔長度 |
| **judge:幻覺** | LLM 判定,見下面 | `--judge` + API key |
| **judge:品質** | LLM 給 1–5 分,見下面 | `--judge` + API key |

ASR 層和潤稿層**用同一組**。算不出來的一律回 `null`,表上印 `—` ——
**「沒量到」不會被寫成 0。**

RAM / CPU / 溫度不在 `grade.py` 裡,那是 `monitor.py` 的事(見最後一節)。

## CER 怎麼算

```
CER = edit_distance(canon(ref), canon(hyp)) / len(canon(ref))
```

`canon()` 在 `typeless/norm.py`,**只有這一份**,ASR 和潤稿共用。
沒有這個共用,兩層會各自漂移,數字就不能比。它做四件事:

1. **NFKC 正規化** —— 全形轉半形,`Ｂ` 變 `B`
2. **繁簡統一** —— `zhconv.convert(..., 'zh-tw')`。ASR 輸出簡體不該被算成錯字,
   那是下游潤稿要處理的事,不是聽錯
3. **去標點、轉小寫** —— 標點是潤稿加的,不是 ASR 聽到的
4. **中文數字轉阿拉伯數字** —— `零點三` → `0.3`、`二十五` → `25`、`一零二四` → `1024`

第 4 點以前沒做,害歷史 CER 全部高估約 6 個百分點:人工寫「零點三」、
ASR 輸出「0.3」,整段被算成錯。修完 whisper 從 15.8% 變 9.8%、
Fun-ASR 8.7% 變 3.5%。**排名沒變,但絕對值差很多。**

`點` 只有在兩邊都是數字時才當小數點 —— 不然「兩點半」會變成「2半」。

## judge 怎麼算(你要知道的部分)

一次 LLM 呼叫,同時看兩件事。輸入是**原文 + 潤稿輸出**,不是只有輸出 ——
沒有原文就分不出「掰的」和「本來就有的」。

`typeless/judge.py` 裡的 `JUDGE_PROMPT` 就是完整判準,可以直接讀。要點:

**先分類。** 輸出是「清理過的逐字稿」還是「重寫」?如果逐字稿裡的人說了
「幫我轉成精簡的 plan」而模型真的照做了 —— 那是重寫,**品質最高只能給 2 分**,
而且要報一個 high 幻覺。這是最重要的一種失敗:模型執行了逐字稿裡的指令,
但那句話是說給別人聽的,不是說給潤稿模型聽的。

**幻覺** = 輸出裡有、而原文支撐不了的內容。按意思判,不是按字面:

| 這些**不算**幻覺(明確列在 prompt 裡) | 這些**算** |
|---|---|
| 大小寫、空白、標點(`bm 25` → `BM25`) | 憑空多出來的總結句 |
| 明顯的拼字修正(`anth` → `Anthropic`) | 把兩句合併成一個新主張 |
| 同音字修正(`形式裡` → `行事曆`,上下文是 Google Calendar) | 改變誰對誰做了什麼 |
| 繁簡轉換、口說數字轉阿拉伯數字 | 否定變肯定 |
| 刪掉語助詞、口吃、重複字 | 刪掉說話者真的講過的內容 |

`severity` 只有 `high`(會改變讀者以為說話者說了什麼)和 `low`。
prompt 明確寫「拿不準就不要報,誤報比漏報糟」。

**品質 1–5:**

| 分 | 意思 |
|---|---|
| 5 | 乾淨、意思完整保留、可以直接貼出去用 |
| 4 | 意思保留,還有少量口語殘留或標點問題 |
| 3 | 可用但明顯不乾淨,或有小的語意偏移 |
| 2 | 明確的語意問題、刪掉實質內容、或整份是重寫 |
| 1 | 大量幻覺,或跟原文關係很小 |

**設定 —— OpenAI 相容,不綁供應商:**

```bash
JUDGE_BASE_URL   預設 https://api.openai.com/v1
JUDGE_MODEL      預設 gpt-5.4-mini
JUDGE_API_KEY    沒設就退回 OPENAI_API_KEY
JUDGE_URL        完整 endpoint 覆寫,給路徑不標準的供應商
```

只要對方吃 `POST {base_url}/chat/completions` 就能用 ——
vLLM、Ollama、groq、together、本機 llama-server 都可以。MiniMax 的路徑不標準,
用 `JUDGE_URL=https://api.minimax.io/v1/text/chatcompletion_v2`。

`temperature=0`。model 名稱與 prompt 的 sha8 會寫進結果 —— **換了裁判,舊分數不可比。**
`max_completion_tokens` 被拒的話自動退回 `max_tokens`;不收 `temperature` 的
reasoning model 會自動拿掉重試。

判定失敗(沒 key、API 掛掉)時 `error` 有值,**那一題不列入統計**,
表上另外報「N 題判定失敗」。不會假裝成 0 個幻覺。

## 怎麼跑

```bash
# 0. 先起服務(這些腳本不負責啟動模型)
./run_qwen35.sh &                        # llama-server :8902
./run_breeze_server.sh &                 # breeze :8380
/usr/bin/python3 funasr_server.py &      # funasr :8379(3.13 拿掉了 cgi,要用 3.9)

# 1. ASR
uv run python asr.py --engine breeze --out asr/breeze.jsonl
#   引擎:whisper / whisper:no-vad / funasr:q4km / breeze / scribe

# 2. 潤稿
uv run python polish.py --asr asr/breeze.jsonl --out polish/breeze-qwen35-v2.jsonl

# 3. 算分
uv run python grade.py --asr asr/breeze.jsonl --polish polish/breeze-qwen35-v2.jsonl \
                       --ref evalset/ --judge --per-item --out result.json

# 要量速度的話,把 1 跟 2 包在 monitor 裡
uv run python monitor.py --out logs/run.jsonl -- \
    uv run python polish.py --asr asr/breeze.jsonl --out polish/x.jsonl
```

`asr.py` / `polish.py` 都是**逐筆寫檔**,中途掛掉不會全白跑,
重跑會自動跳過已完成的 id(`--overwrite` 可覆蓋)。

`grade.py` 讀到 polish 檔裡已經有 `judge` 欄位就直接用,不會重打 API。
判完想留著的話把結果併回 JSONL 即可。

交換格式看 [`FORMATS.md`](./FORMATS.md)。任何 pipeline 只要把結果轉成那個格式就能評。

## monitor.py —— 速度數字可不可信

同一個模型在這台 16GB 機器上量過 **27.5 tok/s** 和 **10.1 tok/s**,差 2.7 倍,
差別完全來自別的程式在搶記憶體頻寬。沒有這份紀錄,就分不出
「模型變快了」和「剛好那次機器比較閒」。

免 sudo 量得到:
- `memory_pressure -Q` 的可用記憶體百分比
- `vm.swapusage` 的 swap 用量,以及**期間 swapout 次數的增量**(這個最準)
- `iostat` 的 CPU user/sys/idle 與 load average
- `pmset -g therm` 的降頻警告
- `ps -r` 最吃 CPU 的前幾名(事後回答「那次是誰在搶」)

**攝氏溫度要 sudo**(`powermetrics` 的限制,不是這支腳本的):

```bash
sudo uv run python monitor.py --temp --out logs/m.jsonl -- <指令>
```

⚠️ 不要看 `vm_stat` 的 "Pages free"。macOS 會把閒置記憶體全部拿去當快取,
那個數字在正常機器上也永遠接近 0,不代表不夠用。

判定標準寫死在 `verdict()` 裡,不是每次憑感覺講:
load 1m 平均 >4、CPU idle 平均 <40%、可用記憶體 <15%、期間有 swapout、
或有非本專案的程式吃 CPU >50% —— 任一條成立就標記速度不可信,並列出原因。

## 目錄

```
asr.py polish.py grade.py monitor.py     ← 主線,就這四支
typeless/
  norm.py         正規化 + CER(唯一一份)
  judge.py        LLM judge(OpenAI 相容)
  engines.py      llama-server client,polish.py 用
  asr_engines.py  ASR 引擎驅動,asr.py 用
evalset/          資料集(manifest.jsonl + audio/ + text/)
prompts/          潤稿 prompt
```

## 現在量不到的東西

| 缺什麼 | 影響 |
|---|---|
| `manifest.gold_final` 全部是 null | **潤稿層的 CER 量不到**。品質只能靠 judge |
| 5/11 的 `.zh-tw.txt` 只做過 zhconv 沒人工修 | 那 5 題的 ASR CER 量到的是「跟 ElevenLabs 差多少」 |
| `teach-03a/b/c` 共用母檔逐字稿 | 這 3 題完全沒有 ASR CER |

補標註**不用重跑模型**。hypothesis 在 `asr/` 和 `polish/`,reference 在 `evalset/`,
靠 id 對接。改完 reference 重跑一次 `grade.py` 就好,幾秒。
