# 交換格式 —— 外部 pipeline 怎麼接進來評分

`./tl` 會自己去跑模型、去 OpenWhispr 的 SQLite 撈結果。
**`grade.py` 不會。** 它只讀你給的檔案路徑,不連任何東西(唯一例外是 `--judge`)。

所以你自己的 pipeline 要接進來,只有一件事要做:**把結果轉成下面的 JSONL。**
轉檔在你那邊做,這支不猜、不掃描、不自動抓。

```bash
uv run python grade.py --asr asr/whisper.jsonl \
                       --polish polish/mine.jsonl \
                       --ref evalset/ \
                       --per-item --out result.json
```

`--asr` 跟 `--polish` 至少要給一個,兩個都給就兩層一起評。

---

## 檔案格式

三個檔案都是 **JSONL(一行一個 JSON object)**,用 `id` 對接。
`id` 就是你自己的 clip 識別字,只要三個檔一致就好。

### `--asr` —— ASR 逐字稿

```json
{"id": "teach-01", "text": "呃 我們來看一下 B M 二十五 的分數", "dur_s": 106.76, "elapsed_s": 12.4}
```

| 欄位 | 必填 | 說明 |
|---|---|---|
| `id` | ✅ | |
| `text` | ✅ | ASR 輸出的逐字稿 |
| `dur_s` | | 音檔秒數。沒有就算不出 RTF |
| `elapsed_s` | | ASR 花的秒數。沒有就算不出 RTF |

**這個檔同時也是潤稿層的 input。** 潤稿的一半指標(語助詞移除率、口吃移除率、
簡體移除率、長度比、幻覺率)是拿 output 跟 input 比出來的 —— 沒有 input 就全是 `—`。

### `--polish` —— 潤稿輸出

```json
{"id": "teach-01", "text": "我們來看一下 BM25 的分數。",
 "latency_s": 15.0, "gen_tok": 376, "tok_s": 27.1, "prompt_tok": 890}
```

| 欄位 | 必填 | 說明 |
|---|---|---|
| `id` | ✅ | |
| `text` | ✅ | 潤完的文字 |
| `input` | | 這一題真正送進模型的文字。省略就用 `--asr` 同 id 的 `text` |
| `latency_s` / `gen_tok` / `tok_s` / `prompt_tok` | | 速度。缺就不報 |
| `judge` | | 已經在別處算好的漂移結果,格式同 `--judge` 的輸出。有就不重打 API |

如果你的 pipeline 在 ASR 跟 LLM 之間還做了別的事(切句、標點還原、
前處理),把**實際送進 LLM 的那份**寫進 `input`。否則移除率會算錯 ——
分母不是你以為的那個分母。

### `--ref` —— 正解(全部選填)

```json
{"id": "teach-01",
 "asr_ref": "呃 我們來看一下 BM25 的分數",
 "polish_ref": "我們來看一下 BM25 的分數。",
 "terms": [["BM25", "B M 25"], ["int8"]],
 "dur_s": 106.76}
```

| 欄位 | 給了才算得出 |
|---|---|
| `asr_ref` | ASR 層的 CER、長度比、術語召回 |
| `polish_ref` | 潤稿層的 `cer_ref`、`len_ratio_ref`、術語保留 |
| `terms` | 術語召回 / 保留。每一項是一組**等價寫法**,任一命中就算有 |
| `dur_s` | RTF |

沒給的欄位,對應指標回 `—`(不是 0)。表最下面的「覆蓋率」會告訴你幾題有正解。

`--ref` 也可以直接指 `evalset/` 目錄 —— 有 `manifest.jsonl` 就自動讀。
注意 `text/<id>.zh-tw.txt` 掛在 **`asr_ref`**,不是 `polish_ref`:那是逐字稿,
語助詞還在。拿它當潤稿正解會讓指標整個反過來(什麼都不刪的模型 CER 最低)。

---

## 懶人格式:一個資料夾裝 `<id>.txt`

`--asr` / `--polish` / `--ref` 都可以直接給一個資料夾,裡面是 `<id>.txt`:

```
asr_out/teach-01.txt   agent-01.txt   …
```

跟 JSONL 等價,只是沒有 `dur_s` / `elapsed_s` / 速度那些欄位(對應指標會是 `—`)。
`--ref` 用資料夾模式時,內容一律當 **`polish_ref`**。

---

## 輸出

stdout 一張表;`--out x.json` 存完整明細,結構是:

```
{
  "asr":    {"items": [{"id", "out", "metrics"}, …], "aggregate": {…}},
  "polish": {"items": [{"id", "raw", "out", "metrics"}, …], "aggregate": {…}},
  "coverage": {"polish_input": 15, "asr_ref": 12, "polish_ref": 0}
}
```

`metrics` 的每個欄位是什麼,看 [`METRICS.md`](./METRICS.md)。

---

## 離線

`grade.py` 預設**完全不連網**。唯一的例外是 `--judge`(語意漂移,
LLM-as-a-judge,需要 `MINIMAX_API_KEY`)。沒設金鑰時它不會假裝成功 ——
會報「N 題判定失敗(不列入)」,不會把「沒量到」寫成 0 drift。
