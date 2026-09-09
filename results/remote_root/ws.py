import numpy as np, torch, gzip, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cut.behavior_battery import entropy_score
from src.cut.analyze_trajectory import auc
dev="cuda"; M="EleutherAI/pythia-1.4b"
tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16).to(dev).eval()
tr=torch.load("results/traj_1.4b.pt",weights_only=False)
n=tr["unemb_cnorm"].shape[1]
r64=tr["resid_i"][tr["steps"].index(64)].numpy()
dec=[tok.decode([i]) for i in range(n)]
ver=np.zeros(n,int)
for l in gzip.open("external/pythia_6_9b.jsonl.gz","rt",encoding="utf-8"):
    r=json.loads(l); i=int(r["i"])
    if i<n and "verified" in r.get("magikarp",""): ver[i]=1
rng=np.random.default_rng(0)
ws=[i for i in range(n) if dec[i].strip()=="" and len(dec[i])>0]
hw=[i for i in ws if r64[i]>=1e-8]        # whitespace that DID get gradient
nw=[i for i in range(n) if dec[i].strip()!="" and r64[i]>=1e-8]
print(f"whitespace tokens total {len(ws)}; of those, received gradient by step64: {len(hw)}")
G={"whitespace, got gradient":hw[:200],
   "whitespace, stillborn":[i for i in ws if r64[i]<1e-8][:200],
   "non-ws, got gradient (healthy)":[int(x) for x in rng.choice(nw,300,replace=False)]}
print(f"\n{'group':>32} | {'n':>4} | {'entropy':>8}")
print("-"*52)
E={}
for g,ids in G.items():
    if not ids: continue
    e=entropy_score(model,tok,dev,ids); E[g]=e
    print(f"{g:>32} | {len(ids):4d} | {e.mean():8.3f}")
a=np.concatenate([E["whitespace, got gradient"],E["whitespace, stillborn"]])
b=E["non-ws, got gradient (healthy)"]
y=np.array([1]*len(a)+[0]*len(b))
print(f"\nentropy AUC: ANY whitespace vs non-whitespace = {auc(np.concatenate([a,b]),y):.3f}")
print("-> if this is high, GlitchMiner's entropy signal is largely a whitespace detector")
