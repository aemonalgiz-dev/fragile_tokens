import numpy as np, torch, gzip, json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.cut.behavior_battery import entropy_score
dev="cuda"; M="EleutherAI/pythia-1.4b"
tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(dev).eval()
V=model.get_input_embeddings().weight.shape[0]
ver=np.zeros(V,int); cat={}; dec={}
for l in gzip.open("external/pythia_6_9b.jsonl.gz","rt",encoding="utf-8"):
    r=json.loads(l); i=int(r["i"])
    if i<V:
        cat[i]=r.get("category"); dec[i]=r.get("decoded")
        if "verified" in r.get("magikarp",""): ver[i]=1
rng=np.random.default_rng(0)
groups={
 "verified glitch":[i for i in range(V) if ver[i]==1],
 "UNREACHABLE_MULTI":[i for i in range(V) if cat.get(i)=="UNREACHABLE_MULTI_TOKEN"][:300],
 "UNDECODEABLE":[i for i in range(V) if cat.get(i)=="UNDECODEABLE"][:300],
 "random OK":[int(x) for x in rng.choice([i for i in range(V) if cat.get(i)=="OK"],300,replace=False)],
}
print(f"{'group':>20} | {'n':>4} | {'mean entropy':>13}")
print("-"*44)
for g,ids in groups.items():
    e=entropy_score(model,tok,dev,ids)
    print(f"{g:>20} | {len(ids):4d} | {e.mean():13.3f}")
# is the verified set dominated by word-fragments the model can complete?
frag=[i for i in groups["verified glitch"] if dec.get(i) and dec[i][:1].isalpha() and not dec[i].startswith(" ")]
print(f"\nverified tokens that are bare word-fragments (no leading space, alphabetic): "
      f"{len(frag)}/{len(groups['verified glitch'])}")
print("  e.g.", [dec[i] for i in frag[:10]])
