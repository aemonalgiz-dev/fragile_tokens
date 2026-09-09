"""One model loader for the scale ladder.

The experiments so far loaded a 7B model with three lines and moved it to one
GPU. Models of 14-32B need bf16 and device_map sharding, a 70B needs 4-bit, and
some families need trust_remote_code. Every script in the ladder takes the same
flags and calls load_from_args, so a model that loads in one experiment loads
in all of them.

With device_map the model must not be .to()'d afterwards; inputs go to the
device of the first parameter, which is where the embedding layer sits.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_model_args(ap):
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto",
                    help="auto = bf16 where the GPU supports it, else fp16")
    ap.add_argument("--load-4bit", action="store_true",
                    help="bitsandbytes NF4; needed for 70B on one 80GB card. A caveat "
                         "for any behavioural comparison, and reported as such.")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--batch", type=int, default=0,
                    help="override the script's batch size (0 = script default)")
    return ap


def load_from_args(a):
    return load(a.model, dtype=a.dtype, load_4bit=a.load_4bit,
                trust_remote_code=a.trust_remote_code)


def load(name, dtype="auto", load_4bit=False, trust_remote_code=False, force_cpu=False):
    """force_cpu (or env GLITCH_FORCE_CPU=1) keeps a smoke test off a GPU that
    happens to be present -- the development machine has one that must not be
    used for this project's compute."""
    import os
    force_cpu = force_cpu or os.environ.get("GLITCH_FORCE_CPU") == "1"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": trust_remote_code}
    if torch.cuda.is_available() and not force_cpu:
        if dtype == "bf16" or (dtype == "auto" and torch.cuda.is_bf16_supported()):
            kw["dtype"] = torch.bfloat16
        else:
            kw["dtype"] = torch.float16
        kw["device_map"] = "auto"
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
            kw.pop("dtype", None)
    else:
        kw["dtype"] = torch.float32
    # device_map needs accelerate; without it, load on CPU and move the whole
    # model to the GPU, which is fine for anything that fits on one card.
    try:
        import accelerate  # noqa: F401
    except ImportError:
        kw.pop("device_map", None)
        if load_4bit:
            raise SystemExit("--load-4bit needs accelerate and bitsandbytes installed")
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except TypeError:                       # transformers < 5 spells it torch_dtype
        if "dtype" in kw:
            kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if "device_map" not in kw and torch.cuda.is_available() and not force_cpu:
        model = model.to("cuda")
    model.eval()
    dev = next(model.parameters()).device
    return model, tok, dev


def load_embeddings(name, trust_remote_code=False):
    """Input embedding, output embedding, final-norm gain, and the tied flag,
    read from the safetensors shards without instantiating the model.

    fragility_predict.py used to load the whole model in fp32 to reach two
    matrices; at 32B that is 128 GB of host memory for 2 GB of data. Falls back
    to a full bf16 CPU load only if the checkpoint is not safetensors.
    Returns (E_in, E_out, gain, tied) as float32 CPU tensors.
    """
    import json, os
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
    try:
        idx = json.load(open(hf_hub_download(name, "model.safetensors.index.json")))
        wmap = idx["weight_map"]
    except Exception:
        try:
            hf_hub_download(name, "model.safetensors")
            from safetensors import safe_open
            f = safe_open(hf_hub_download(name, "model.safetensors"), "pt")
            wmap = {k: "model.safetensors" for k in f.keys()}
        except Exception:
            wmap = None
    if wmap is None:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                     trust_remote_code=trust_remote_code)
        E_in = model.get_input_embeddings().weight.detach().float()
        E_out = model.get_output_embeddings().weight.detach().float()
        g = torch.ones(E_in.shape[1])
        for path in ("model.norm", "gpt_neox.final_layer_norm", "transformer.ln_f"):
            o, ok = model, True
            for p in path.split("."):
                if not hasattr(o, p):
                    ok = False; break
                o = getattr(o, p)
            if ok and getattr(o, "weight", None) is not None:
                g = o.weight.detach().float(); break
        return E_in, E_out, g, bool(getattr(cfg, "tie_word_embeddings", False))
    from safetensors import safe_open

    def find(cands):
        for k in wmap:
            if any(c in k for c in cands):
                return k
        return None
    k_in = find(["embed_tokens.weight", "embed_in.weight", "wte.weight", "word_embeddings.weight"])
    k_out = find(["lm_head.weight", "embed_out.weight"])
    k_norm = None
    for k in wmap:
        if k.endswith(("model.norm.weight", "final_layer_norm.weight", "ln_f.weight",
                       "final_layernorm.weight", "transformer.norm.weight")) and "layers." not in k:
            k_norm = k; break

    def get(k):
        f = safe_open(hf_hub_download(name, wmap[k]), "pt")
        return f.get_tensor(k).float()
    E_in = get(k_in)
    tied = k_out is None or bool(getattr(cfg, "tie_word_embeddings", False))
    E_out = E_in.clone() if k_out is None else get(k_out)
    g = get(k_norm) if k_norm else torch.ones(E_in.shape[1])
    return E_in, E_out, g.flatten()[:E_in.shape[1]], tied


def describe(model, tok, name):
    cfg = model.config
    p = next(model.parameters())
    return {"model": name, "commit": getattr(cfg, "_commit_hash", "unknown"),
            "dtype": str(p.dtype), "device": str(p.device),
            "num_layers": getattr(cfg, "num_hidden_layers", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "vocab_size": model.get_input_embeddings().weight.shape[0],
            "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
            "architecture": (getattr(cfg, "architectures", None) or ["?"])[0],
            "quantized": hasattr(cfg, "quantization_config"),
            "tokenizer_class": type(tok).__name__}
