# ASR 層基準 — 2026-08-25

四個引擎 × evalset 14 個 clip,全部直接呼叫引擎(沒走 OpenWhispr app)。
產生指令在每一節下面,原始輸出在 `asr/*.jsonl`(不進版控)。

## CER(越低越好)

`✓` = ref 是人工修過的  `~` = ref 只做過 zhconv,沒人工修  `·` = 還沒有 ref

| clip | | whisper-turbo | funasr-q4km | breeze | scribe |
|---|---|---:|---:|---:|---:|
| agent-01 | ✓ | 12.8% | 13.2% | 10.7% | 11.3% |
| agent-02 | ✓ | 14.2% | 10.6% | 7.7% | **5.1%** |
| agent-03 | ✓ | 11.9% | 11.7% | 8.5% | **4.0%** |
| agent-04 | ✓ | 10.7% | 9.0% | 5.9% | **5.8%** |
| agent-05 | ✓ | 18.2% | 4.1% | 6.4% | **1.4%** |
| meeting-01 | ~ | 11.1% | 11.1% | 8.9% | 11.1% |
| meeting-03 | ✓ | 33.6% | 29.1% | 23.6% | 22.3% |
| meeting-04 | ~ | 14.9% | 11.5% | 8.0% | 5.9% |
| meeting-05 | ~ | 34.0% | 32.3% | 23.5% | 14.8% |
| teach-01 | ~ | 19.8% | 22.8% | 10.5% | 8.1% |
| teach-02 | ~ | 10.0% | 19.7% | 9.1% | 7.2% |
| teach-03a/b/c | · | — | — | — | — |

| 取樣 | whisper-turbo | funasr-q4km | breeze | scribe |
|---|---:|---:|---:|---:|
| 有 ref 全部(11) | 17.38% | 15.91% | **11.17%** | 8.82%* |
| 只算人工修過(6) | 16.91% | 12.96% | **10.46%** | 8.31%* |
| 只算未修(5) | 17.95% | 19.46% | **12.02%** | 9.42%* |

\* **scribe 的數字不能跟其他三個並排比。** ref 就是 scribe(ElevenLabs)產的,
人工只改了一部分。它的 CER 量到的是「人工當初改了多少字」,是下界不是實力。
在只算未修的那 5 個上它是 9.42% —— 那 5 個 ref 本來就是它自己的輸出轉繁而已,
理論值應該接近 0,9.42% 全部來自繁簡轉換與正規化的殘差。

**meeting-03 / meeting-05 四個引擎全部 >22%**,不是引擎爛,是那兩段音檔本身難
(meeting-05 是 298s 的長會議)。要判斷引擎優劣看其他九個。

## RTF(處理秒數 / 音檔秒數,越小越快)

| 引擎 | 平均 | 最慢 |
|---|---:|---:|
| scribe | 0.070 | 0.149 |
| funasr-q4km | 0.073 | 0.225 |
| whisper-turbo | 0.083 | 0.227 |
| breeze | 0.121 | 0.247 |

全部遠低於 1,即時性都不是瓶頸。breeze 最慢但也只有 0.12。
⚠️ 這是**獨佔機器**跑出來的;16GB 上同時開別的東西會整組變慢。

## 簡體殘留(全 14 個 clip 加總)

| 引擎 | 字數 |
|---|---:|
| breeze | **2** |
| whisper-turbo | 400 |
| funasr-q4km | 1687 |
| scribe | 1883 |

breeze 是唯一直接輸出繁體的。其他三個要靠下游潤稿轉,或先過 zhconv。

## 長度比(輸出字數 / ref 字數,正規化後)

抓「整句被吞掉」用的護欄。CER 對「少了一句」只會小幅上升,長度比會直接掉下來。

whisper 有四個 clip 掉到 0.83 以下:agent-05 0.832、meeting-03 **0.770**、
meeting-05 0.829、agent-02 0.905。funasr / breeze / scribe 都在 0.91–1.07。

## ⚠️ 最大的發現:whisper 的 VAD 會吃掉安靜的錄音

teach-03b / teach-03c 的 mean volume 是 **-51 dB**(其他 clip 約 -31 到 -47)。

字數/秒(正常值 4.0–6.5):

| clip | whisper +VAD | whisper 無VAD | funasr +VAD | funasr 無VAD | breeze | scribe |
|---|---:|---:|---:|---:|---:|---:|
| teach-03b | **0.6** | 2.5 | **1.2** | 6.2 | 3.5 | 3.3 |
| teach-03c | **0.6** | 3.6 | **2.0** | 5.0 | 4.2 | 4.9 |

whisper 開 VAD 在這兩段只吐出 ~10% 的內容。Silero 的 threshold 0.5 把 -51 dB
的語音當成靜音丟掉了。這組 VAD 參數是抄 OpenWhispr 的,所以**產品上會真的發生**。

## VAD 開/關不是單純的好壞

| 取樣 | whisper +VAD | whisper 無VAD | funasr +VAD | funasr 無VAD |
|---|---:|---:|---:|---:|
| 有 ref 全部(11) | **17.38%** | 17.99% | **15.91%** | 51.55% |
| 只算人工修過(6) | **16.91%** | 17.48% | **12.96%** | 49.01% |

- whisper:整體幾乎沒差(+0.6),但安靜的檔案關掉 VAD 明顯較好。
- funasr:**關掉 VAD 直接崩掉**(+35.6 個百分點)。長音檔沒有切段就會亂跑 ——
  agent-04 96.3%、meeting-05 93.7%。funasr 的 VAD 是必要的,不是可選的。

**結論:預設維持 VAD 開**(也跟 OpenWhispr 的上傳路徑一致)。
但安靜錄音要另外處理,不然 whisper 會靜靜地少掉九成內容。

## 怎麼重現

```bash
# 先起 server(funasr 要用 /usr/bin/python3,3.13 拿掉了 cgi)
/usr/bin/python3 funasr_server.py &      # :8379
./run_breeze_server.sh &                 # :8380

uv run python asr.py --engine whisper     --out asr/whisper-turbo.jsonl
uv run python asr.py --engine funasr:q4km --out asr/funasr-q4km.jsonl
uv run python asr.py --engine breeze      --out asr/breeze.jsonl
uv run python asr.py --engine scribe      --out asr/scribe.jsonl

uv run python grade.py --asr asr/whisper-turbo.jsonl --ref evalset/ --per-item
```

總耗時約 8 分鐘(四個引擎 × 14 檔)。

## 還缺什麼

| 缺的東西 | 影響 |
|---|---|
| `teach-01 / teach-02 / meeting-01 / meeting-04 / meeting-05` 的 ref 還沒人工修 | 這 5 個現在等於在量「跟 ElevenLabs 差多少」,不是「跟正確答案差多少」 |
| `teach-03a/b/c` 沒有各自的逐字稿(共用母檔) | 完全沒有 CER,只能看字數/秒 |
| `manifest.terms` 全部是空的 | 術語召回量不了 —— 這是這個任務最該量的東西之一 |
| `gold_final` 全部是 null | 潤稿層的 CER / 術語保留量不了 |

**補標註不用重跑 ASR。** hypothesis 存在 `asr/*.jsonl`,ref 在 `evalset/`,
兩邊靠 id 對接。改完 ref 只要重跑一次 `grade.py`(幾秒)。
