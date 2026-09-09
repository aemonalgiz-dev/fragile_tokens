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
sb=[i for i in range(n) if r64[i]<1e-8]
sb_ws=[i for i in sb if dec[i].strip()==""]
sb_nw=[i for i in sb if dec[i].strip()!=""]
rng=np.random.default_rng(0)
hl=[int(x) for x in rng.choice([i for i in range(n) if r64[i]>=1e-8],400,replace=False)]
hl_nw=[i for i in hl if dec[i].strip()!=""][:300]
print(f"STILLBORN defined by gradient (zero token-specific input update @step64): {len(sb)}")
print(f"   whitespace {len(sb_ws)} | NON-whitespace {len(sb_nw)}")
print("   non-ws examples:", [repr(dec[i])[:16] for i in sb_nw[:14]])
print()
groups={"stillborn NON-whitespace":sb_nw[:300],"stillborn whitespace":sb_ws[:300],
        "healthy non-whitespace":hl_nw}
E={}
print(f"{'group':>26} | {'n':>4} | {'mean entropy':>13}")
print("-"*50)
for g,ids in groups.items():
    if not ids: continue
    e=entropy_score(model,tok,dev,ids); E[g]=e
    print(f"{g:>26} | {len(ids):4d} | {e.mean():13.3f}")
if "stillborn NON-whitespace" in E:
    a=E["stillborn NON-whitespace"]; b=E["healthy non-whitespace"]
    y=np.array([1]*len(a)+[0]*len(b))
    print(f"\nentropy AUC, NON-whitespace stillborn vs healthy: {auc(np.concatenate([a,b]),y):.3f}")
    print("-> if this stays high, the stillborn/entropy link is not a whitespace artifact")
