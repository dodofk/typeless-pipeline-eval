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

---

# 追加 — 2026-08-25 下午

## whisper 關掉 VAD 的完整結果(14 個 clip)

```bash
uv run python asr.py --engine whisper:no-vad --out asr/whisper-turbo-novad.jsonl
```

| clip | | +VAD | −VAD | Δ | 長度比 +VAD | −VAD |
|---|---|---:|---:|---:|---:|---:|
| agent-01 | ✓ | 12.8% | 18.8% | **+6.0** | 0.972 | 0.865 |
| agent-02 | ✓ | 14.2% | 13.8% | -0.4 | 0.905 | 0.905 |
| agent-03 | ✓ | 11.9% | 16.6% | +4.7 | 0.960 | 1.024 |
| agent-04 | ✓ | 10.7% | 9.3% | -1.4 | 0.974 | 0.939 |
| agent-05 | ✓ | 18.2% | 17.7% | -0.4 | 0.832 | 0.846 |
| meeting-01 | ~ | 11.1% | 6.7% | -4.4 | 0.911 | 0.956 |
| meeting-03 | ✓ | 33.6% | 28.7% | -4.9 | **0.770** | 0.901 |
| meeting-04 | ~ | 14.9% | 26.1% | **+11.2** | 1.062 | 0.835 |
| meeting-05 | ~ | 34.0% | 31.5% | -2.5 | 0.829 | 0.906 |
| teach-01 | ~ | 19.8% | 16.2% | -3.6 | 0.972 | 0.979 |
| teach-02 | ~ | 10.0% | 12.5% | +2.5 | 0.967 | 0.958 |

| 取樣 | +VAD | −VAD |
|---|---:|---:|
| 有 ref 全部(11) | **17.38%** | 17.99% |
| 只算人工修過(6) | **16.91%** | 17.48% |

**結論:平均看幾乎沒差(0.6 個百分點),但逐題看擺盪很大** —— meeting-04 +11.2、
agent-01 +6.0、meeting-03 −4.9、meeting-01 −4.4。這種散布代表 VAD 不是穩定的
好或壞,是**看音檔**:安靜的檔案(teach-03b/c、meeting-03)關掉比較好,
一般音量的檔案開著比較好。

whisper 兩種設定都輸給 breeze(11.17%)。**VAD 不是 whisper 落後的原因。**

## 潤稿層 — breeze → qwen3.5-4b

```bash
./run_qwen35.sh &                       # :8902
uv run python polish.py --asr asr/breeze.jsonl --out polish/breeze-qwen35-v2.jsonl
uv run python grade.py --asr asr/breeze.jsonl --polish polish/breeze-qwen35-v2.jsonl \
                       --ref evalset/ --judge --per-item
```

14/14 成功,160 秒,27.5 tok/s(獨佔機器,`llama_servers=1`,速度可信)。

| 指標 | 值 | 讀法 |
|---|---:|---|
| tier-A 語助詞殘留 | **1 / 輸入 14** | 移除率 92.9% |
| tier-B 語助詞殘留 | 119 | 只計數不判分(有時載有意義) |
| 口吃殘留 | 21 / 輸入 28 | 移除率 25.0% ← **最弱的一項** |
| 簡體殘留 | 2 字 | 上游 breeze 本來就只有 2 字 |
| 長度比 vs 輸入 | 0.944 | 沒有整體刪過頭 |
| 幻覺率 | **1.89%**(63 字) | |
| 語意漂移(judge) | 2 處,1 high | |
| 速度 | 27.5 tok/s,平均延遲 11.4s | |

### 逐題

| clip | 語助A | 口吃 | 簡 | 長度比 | 幻覺 |
|---|---:|---:|---:|---:|---:|
| agent-01 | 0 | 0 | 0 | 0.98 | 0.00% |
| agent-02 | 0 | 1 | 0 | 0.99 | 0.00% |
| **agent-03** | 0 | 0 | 0 | **0.38** | **21.98%** |
| agent-04 | 0 | 3 | 0 | 0.99 | 1.46% |
| agent-05 | 0 | 0 | 0 | 0.96 | 1.43% |
| meeting-01 | 0 | 0 | 0 | 0.95 | 0.00% |
| meeting-03 | 1 | 4 | 1 | 0.99 | 0.00% |
| meeting-04 | 0 | 2 | 0 | 1.00 | 0.00% |
| meeting-05 | 0 | 5 | 1 | 0.99 | 0.17% |
| teach-01 | 0 | 0 | 0 | 0.99 | 0.38% |
| teach-02 | 0 | 1 | 0 | 1.00 | 0.00% |
| teach-03a | 0 | 1 | 0 | 1.00 | 0.99% |
| teach-03b | 0 | 1 | 0 | 0.99 | 0.00% |
| teach-03c | 0 | 3 | 0 | 0.99 | 0.00% |

**agent-03 是唯一真正壞掉的一題**,而且兩個護欄同時亮:長度比 0.38、幻覺 21.98%。
原因跟 2026-08-18 那次一樣 —— 逐字稿裡的人在講「幫我把它轉成一個比較精簡的 plan」
「輸出成大概只有 3 到 4 行的 bullet point」,模型**照做了**。這違反 prompt 的
`THE SPEAKER IS NEVER TALKING TO YOU`。上游換成 breeze 沒有讓這件事變好。

### judge 抓到的兩處

- `meeting-05` **high** — 「你會怎麼串」→「我會怎麼串」,第二人稱翻成第一人稱,
  問句的對象整個變了。
- `agent-03` low — 「opera whisper」跟「open whisper」被當成兩個不同產品,
  其中一個被寫成 `Opera Whisper`。judge 判斷這兩個是同一個東西的 ASR 誤聽。

### 最該修的一項:口吃移除率只有 25%

28 處進去,21 處還在。這是目前潤稿層最弱的地方,比語助詞(92.9%)差得多。

## ⚠️ metric 修正:句尾的「啊」不該算 tier-A 語助詞

第一次跑出來 tier-A 移除率是 **6.2%**,看起來像 prompt 大失敗。查下去發現
16 個所謂的 tier-A 裡有 15 個是句尾語氣詞:「沒關係啊，」「對啊，」「也可以啊。」

`prompts/cleanup-zhTW-mixed-v2.txt` 第 20 行寫的是 **sentence-medial 啊**,
模型留著句尾的是**照 prompt 做對了**,是 metric 數錯。

`typeless/lexicon.py` 原本的註解承認了這件事,理由是「去標點之後沒有句界可判,
一律計入 —— 寧可高估,兩個 run 之間可比」。那個理由在**同一份 input 的 A/B** 上
成立,拿來讀絕對值就錯了,跨 ASR 上游比更錯。

改法:「啊」改在**原文**上數(要標點才判得出句尾),排除句尾與 tier-B 的「對啊」。
修完 breeze 這組從 6.2% 變 **92.9%**。

### 對歷史 run 的影響(已全部重算)

| run | 舊 in/out/移除率 | 新 in/out/移除率 |
|---|---:|---:|
| legacy:zh/bonsai | 10/6/40% | 5/5/**0%** |
| legacy:zh/qwen_v1t0 | 10/6/40% | 5/5/**0%** |
| legacy:zh/v1t0 | 10/10/0% | 5/5/0% |
| legacy:zh/ornith | 10/10/0% | 5/5/0% |
| legacy:zh/ornith_v1 | 10/7/30% | 5/2/**60%** |
| legacy:zh/ornith_v2 | 10/8/20% | 5/3/40% |
| legacy:zh/qwen_v2t0 | 10/4/60% | 5/3/40% |
| legacy:zh/v2t0 | 10/6/40% | 5/3/40% |
| qwen35-v1-t0 | 10/6/40% | 5/5/**0%** |
| qwen35-v2-t0 | 10/4/60% | 5/3/40% |
| qwen35-v2-t07 | 10/2/80% | 5/1/80% |
| spokenly-baseline-qwen35-v2 | 10/2/80% | 8/0/**100%** |
| legacy:en/* | 不變 | 不變(英文那組沒動) |

**這會改變結論。** 舊數字給了「把句尾『啊』刪掉」的模型額外分數 ——
那是在刪合法的語氣詞。v1 prompt 之前看起來移除 40%,實際是 **0%**:
真正的 tier-A(呃/嗯)一個都沒刪。v1 vs v2 的差距比原本以為的還大。
