"""Which base -> post-training stage series are usable for the stage-relative
abandonment test?

A series is usable only if every stage shares one vocabulary (otherwise rows are
not comparable) and, ideally, has untied embeddings (tying gives a rare row
dense softmax-negative gradient every step, which is exactly the sparsity the
mechanism is about).
"""
from __future__ import annotations
import json, urllib.request

FAMILIES = {
    "OLMo2-1B": ["allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT",
                 "allenai/OLMo-2-0425-1B-DPO", "allenai/OLMo-2-0425-1B-Instruct"],
    "OLMo2-7B": ["allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT",
                 "allenai/OLMo-2-1124-7B-DPO", "allenai/OLMo-2-1124-7B-Instruct"],
    "OLMo2-13B": ["allenai/OLMo-2-1124-13B", "allenai/OLMo-2-1124-13B-SFT",
                  "allenai/OLMo-2-1124-13B-DPO", "allenai/OLMo-2-1124-13B-Instruct"],
    "OLMoE": ["allenai/OLMoE-1B-7B-0924", "allenai/OLMoE-1B-7B-0924-SFT",
              "allenai/OLMoE-1B-7B-0924-Instruct"],
    "Tulu3-8B": ["allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO",
                 "allenai/Llama-3.1-Tulu-3-8B"],
    "Amber": ["LLM360/Amber", "LLM360/AmberChat", "LLM360/AmberSafe"],
    "Qwen2.5-7B": ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
    "Qwen2.5-1.5B": ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"],
    "Zephyr": ["mistralai/Mistral-7B-v0.1", "HuggingFaceH4/zephyr-7b-sft-full",
               "HuggingFaceH4/zephyr-7b-beta"],
    "Qwen3-8B": ["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"],
    "Llama3.1-8B": ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
}


def cfg(m):
    u = f"https://huggingface.co/{m}/raw/main/config.json"
    return json.load(urllib.request.urlopen(u, timeout=30))


def main():
    usable = {}
    for fam, models in FAMILIES.items():
        rows = []
        for m in models:
            try:
                c = cfg(m)
                rows.append((m, c.get("vocab_size"), c.get("tie_word_embeddings"),
                             c.get("num_hidden_layers")))
            except Exception as ex:
                rows.append((m, None, None, type(ex).__name__))
        vocabs = {r[1] for r in rows}
        ok = len(vocabs) == 1 and None not in vocabs
        tied = any(r[2] for r in rows)
        flag = "USABLE" if ok else "SKIP  "
        if ok and tied:
            flag = "TIED  "
        print(f"{flag} {fam:14s} vocab={sorted(v for v in vocabs if v)} "
              f"tied={ [r[2] for r in rows] }")
        for m, v, t, n in rows:
            if v is None:
                print(f"         {m}  -> {n}")
        if ok:
            usable[fam] = {"models": models, "vocab": rows[0][1], "tied": tied}
    print()
    print("usable series:")
    for k, v in usable.items():
        print(f"  {k:14s} {len(v['models'])} stages, vocab {v['vocab']}, "
              f"tied={v['tied']}")
    json.dump(usable, open("results/stage_families.json", "w"), indent=1)


if __name__ == "__main__":
    main()
