import numpy as np, torch, gzip, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cut.behavior_battery import entropy_score
from src.cut.analyze_trajectory import auc
dev="cuda"; M="allenai/OLMo-2-0425-1B"
tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(dev).eval()
V=model.get_input_embeddings().weight.shape[0]
ver=np.zeros(V,int); cat={}; dec={}
for l in gzip.open("external/allenai_OLMo_2_1124_7B.jsonl.gz","rt",encoding="utf-8"):
    r=json.loads(l); i=int(r["i"])
    if i<V:
        cat[i]=r.get("category"); dec[i]=r.get("decoded")
        if "verified" in r.get("magikarp",""): ver[i]=1
rng=np.random.default_rng(0)
G={"verified glitch":[i for i in range(V) if ver[i]==1],
   "UNREACHABLE_MULTI":[i for i in range(V) if cat.get(i)=="UNREACHABLE_MULTI_TOKEN"],
   "random OK":[int(x) for x in rng.choice([i for i in range(V) if cat.get(i)=="OK"],400,replace=False)]}
print(f"OLMo-2  ({len(G['verified glitch'])} verified, vs Pythia's 36)\n")
tr=torch.load("results/traj_olmo2_1b.pt", weights_only=False)
n=tr["unemb_cnorm"].shape[1]; i300=tr["steps"].index(300)
print(f"{'group':>20} | {'n':>4} | {'mean entropy':>13} | {'% zero-update @300':>19}")
print("-"*66)
ent={}
for g,ids in G.items():
    sub=ids[:400]
    e=entropy_score(model,tok,dev,sub); ent[g]=e
    r=tr["resid_i"][i300].numpy()[np.array([i for i in sub if i<n], dtype=int)]
    print(f"{g:>20} | {len(sub):4d} | {e.mean():13.3f} | {(r<1e-8).mean():18.1%}")
y=np.array([1]*len(ent["verified glitch"])+[0]*len(ent["random OK"]))
s=np.concatenate([ent["verified glitch"],ent["random OK"]])
print(f"\nentropy AUC (verified vs healthy): {auc(s,y):.3f}")
frag=[i for i in G["verified glitch"] if dec.get(i) and dec[i][:1].isalpha() and not dec[i].startswith(" ")]
print(f"bare word-fragments among verified: {len(frag)}/{len(G['verified glitch'])} = {len(frag)/len(G['verified glitch']):.0%}")
print("  e.g.", [dec[i] for i in frag[:8]])
