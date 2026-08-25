"""[DEPRECATED 2026-08-25] 已被 typeless/ 取代。

    ./tl legacy --which asr      # 把 out/ 的歷史輸出匯成 run
    ./tl report --arm asr        # 印這張表

新舊已逐格比對過(24 個 clip):
  - CER 一致下修約 6 個百分點。舊的 norm() 沒有中文數字轉換,gold 寫「零點三」、
    ASR 輸出「0.3」被算成錯。whisper 15.8%→9.8%、Fun-ASR 8.7%→3.5%。排名不變。
  - 術語召回有 2 格上修(int八→int8、B M二十五→bm25),同一個原因。
  - PUNCT 不再吃掉 '%',所以「78.5%」跟「78.5」現在分得開。

留著當回歸基準,不要當真值用。
"""
import sys, re, pathlib, zhconv
PUNCT = re.compile(r"[\s，。、？！；：「」『』（）,.\?!;:\"'()%\-—…·]+")
def norm(s):
    return PUNCT.sub("", zhconv.convert(s.strip(),'zh-tw').lower())
def cer(ref,hyp):
    r,h=norm(ref),norm(hyp); prev=list(range(len(h)+1))
    for i,rc in enumerate(r,1):
        cur=[i]
        for j,hc in enumerate(h,1): cur.append(min(prev[j]+1,cur[j-1]+1,prev[j-1]+(rc!=hc)))
        prev=cur
    return prev[-1]/max(len(r),1)
K={"long1_meeting":[("baseline",),("model",),("accuracy",),("latency",),("request",),("production",),
   ("batch size","batchsize"),("quantization",),("int8",),("trade-off","tradeoff","trade off"),
   ("七十八點五","78.5"),("一點二","1.2"),("三十二","32"),("六十四","64")],
   "long2_technical":[("react",),("typescript",),("fastapi",),("postgresql",),("redis",),("a100",),("vllm",),
   ("inference",),("throughput",),("token",),("bge-m3","bgem3"),("qdrant",),("top-k","topk"),("hybrid search",),
   ("bm25",),("dense",),("rerank",),("六十","60"),("一零二四","1024"),("零點三","0.3"),("零點七","0.7")],
   "long3_action":[("action item",),("小陳",),("evaluation",),("script",),("push",),("repo",),("amy",),("slack",),
   ("八月二十二","8月22"),("八月二十五","8月25"),("八月二十七","8月27"),("八月二十八","8月28"),
   ("兩點半","2點半"),("十點","10點")]}
def hits(stem,hyp):
    h=norm(hyp); miss=[k[0] for k in K[stem] if not any(norm(a) in h for a in k)]
    return len(K[stem])-len(miss), len(K[stem]), miss
def zh(s):
    t=sum(1 for c in s if c!=zhconv.convert(c,'zh-cn')); return "繁" if t>3 else "簡"
eng=sys.argv[1]
print(f"{'clip':<30}{'CER':>7} {'術語+數字':>9} {'體':>3} {'長度比':>7}  漏掉")
for stem in K:
    ref=pathlib.Path(f"gold/{stem}.txt").read_text().strip()
    for tag in [f"mm28_{stem}_f10",f"mm28_{stem}_m115",f"say_{stem}"]:
        p=pathlib.Path(f"out/{eng}_{tag}.txt")
        if not p.exists(): continue
        hyp=p.read_text(); n,d,m=hits(stem,hyp)
        print(f"{tag:<30}{cer(ref,hyp)*100:6.1f}% {n:>5}/{d:<3} {zh(hyp):>3} {len(norm(hyp))/len(norm(ref)):6.2f}  {','.join(m) if m else '—'}")
