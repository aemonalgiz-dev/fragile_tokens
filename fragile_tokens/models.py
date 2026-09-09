"""Loading models, tokenizers and embedding tables, with the fixes the paper's runs needed for FP8
mixture-of-experts checkpoints."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from .geometry import Embeddings
from .types import CausalLMLike, TokenizerLike

logger = logging.getLogger(__name__)

DType = Literal["auto", "bf16", "fp16", "fp32"]


@dataclass
class LoadedModel:
    """A model ready for scoring, with the device it lives on and provenance for the report."""

    model: CausalLMLike
    tokenizer: TokenizerLike
    device: torch.device
    name: str
    info: dict[str, Any] = field(default_factory=dict)


def load_model(name: str, *, dtype: DType = "auto", load_4bit: bool = False, trust_remote_code: bool = False,
               force_cpu: bool = False, max_gpu_memory: str | None = None) -> LoadedModel:
    """Load a causal language model for teacher-forced scoring.

    Args:
        name: Hugging Face id or local path.
        dtype: ``auto`` picks bfloat16 where supported, float16 otherwise; the CPU path is float32.
        load_4bit: NF4 quantization through bitsandbytes (needs ``accelerate`` and ``bitsandbytes``).
        trust_remote_code: passed through to transformers.
        force_cpu: keep the model off any GPU (also ``FRAGILE_TOKENS_FORCE_CPU=1``).
        max_gpu_memory: per-card cap such as ``"56GiB"``; spreads a model across cards and keeps
            headroom for the fused-expert conversion of large FP8 mixtures.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    force_cpu = force_cpu or os.environ.get("FRAGILE_TOKENS_FORCE_CPU") == "1"
    # The DeepGEMM FP8 linear kernel crashed on H100 with torch 2.14; the Triton fallback is correct.
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    patch_qwen4_exp_indexer()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    use_cuda = torch.cuda.is_available() and not force_cpu
    if use_cuda:
        if dtype == "bf16" or (dtype == "auto" and torch.cuda.is_bf16_supported()):
            kw["dtype"] = torch.bfloat16
        elif dtype == "fp32":
            kw["dtype"] = torch.float32
        else:
            kw["dtype"] = torch.float16
        kw["device_map"] = "auto"
        if max_gpu_memory:
            kw["max_memory"] = {i: max_gpu_memory for i in range(torch.cuda.device_count())}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
            kw.pop("dtype", None)
    else:
        kw["dtype"] = torch.float32
    try:
        import accelerate  # noqa: F401
    except ImportError:
        kw.pop("device_map", None)
        if load_4bit:
            raise RuntimeError("load_4bit needs accelerate and bitsandbytes installed") from None
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except TypeError:  # transformers < 5 spells it torch_dtype
        if "dtype" in kw:
            kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if "device_map" not in kw and use_cuda:
        model = model.to("cuda")
    model.eval()
    n_fixed = dequantize_fp8_embeddings(model, name)
    if n_fixed:
        logger.info("rescaled %d FP8 embedding table(s)", n_fixed)
    device = next(model.parameters()).device
    info = describe_model(model, tok, name)
    logger.info("loaded %s on %s (%s)", name, device, info.get("dtype"))
    return LoadedModel(model=model, tokenizer=tok, device=device, name=name, info=info)


def describe_model(model: Any, tok: Any, name: str) -> dict[str, Any]:
    """Provenance recorded with every scan."""
    cfg = getattr(model, "config", None)
    try:
        import transformers

        tv = transformers.__version__
    except Exception:  # pragma: no cover
        tv = "unknown"
    p = next(model.parameters(), None)
    return {
        "model": name,
        "architecture": type(model).__name__,
        "dtype": str(p.dtype) if p is not None else None,
        "device": str(p.device) if p is not None else None,
        "vocab_size": int(getattr(cfg, "vocab_size", len(tok))) if cfg is not None else len(tok),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_layers": getattr(cfg, "num_hidden_layers", None),
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)) if cfg is not None else None,
        "torch": torch.__version__,
        "transformers": tv,
    }


def patch_qwen4_exp_indexer() -> bool:
    """Qwen3.8-Flash-Next's full-attention layers select keys with a "QSA indexer" whose transformers
    implementation loops in Python over every (batch, query) pair. When the key length is within the
    indexer's budget the selection is the identity, so an all-selected mask is exact; the slow path is
    kept for longer sequences. Returns True when the patch is in place or the class is present."""
    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    except Exception:
        return False
    cls = getattr(m, "Qwen4ExpTextQSAIndexer", None)
    if cls is None or getattr(cls, "_fragile_tokens_fast_path", False):
        return cls is not None
    slow = cls.forward

    def forward(self: Any, hidden_states: torch.Tensor, position_embeddings: Any, attention_mask: torch.Tensor | None,
                past_key_values: Any = None) -> Any:
        if attention_mask is None or attention_mask.shape[-1] > self.token_budget:
            return slow(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        if past_key_values is not None:
            qk = self.index_qk_proj(hidden_states)
            _, token_k = torch.split(
                qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
            raw_keys = token_k.reshape(*hidden_states.shape[:2], self.index_kv_heads, self.index_head_dim).squeeze(2)
            past_key_values.update_indexer(raw_keys, self.layer_idx)
        if attention_mask.dtype == torch.bool:
            return torch.ones_like(attention_mask)
        return torch.zeros_like(attention_mask)

    cls.forward = forward
    cls._fragile_tokens_fast_path = True
    logger.info("qwen4_exp QSA indexer: exact fast path for key lengths within the budget")
    return True


def dequantize_fp8_embeddings(model: Any, name: str) -> int:
    """FP8 checkpoints can quantize an ``nn.Embedding`` table with one per-tensor scale that
    transformers drops on load, leaving raw FP8 codes in the lookup. Re-attach the scale as a forward
    hook. Returns the number of tables patched."""
    fp8 = {torch.float8_e4m3fn, torch.float8_e5m2}
    embs = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Embedding) and m.weight.dtype in fp8]
    if not embs:
        return 0
    from safetensors import safe_open

    if Path(name).is_dir():
        d = Path(name)
    else:
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(name, allow_patterns=["*.json"]))
    wm = json.loads((d / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    for n, m in embs:
        tail = n.split("layers.", 1)[1] if "layers." in n else n
        cands = [k for k in wm if k.endswith("." + tail + ".weight_scale") or k.endswith("." + tail + ".weight_scale_inv")]
        if len(cands) != 1:
            raise RuntimeError(f"FP8 embedding {n}: expected one scale key in the checkpoint, found {cands}")
        with safe_open(str(d / wm[cands[0]]), "pt") as f:
            scale = f.get_tensor(cands[0]).float()
        if scale.numel() != 1:
            raise RuntimeError(f"FP8 embedding {n}: only a per-tensor scale is handled, got {tuple(scale.shape)}")
        sc = float(scale.item())
        m.register_forward_hook(lambda mod, inp, out, sc=sc: out.to(torch.bfloat16) * sc)
        logger.info("fp8 embedding %s: lookup rescaled by %.4g", n, sc)
    return len(embs)


def load_embeddings(name: str, trust_remote_code: bool = False) -> Embeddings:
    """Input embedding, output embedding (with the final-norm gain folded in) and the tied flag, read
    from the safetensors shards without instantiating the model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
    wmap: dict[str, str] | None
    try:
        wmap = json.loads(Path(hf_hub_download(name, "model.safetensors.index.json")).read_text(encoding="utf-8"))["weight_map"]
    except Exception:
        try:
            single = hf_hub_download(name, "model.safetensors")
            with safe_open(single, "pt") as f:
                wmap = {k: "model.safetensors" for k in f.keys()}
        except Exception:
            wmap = None
    if wmap is None:
        return _embeddings_from_model(name, cfg, trust_remote_code)

    def find(cands: list[str]) -> str | None:
        hits = [k for k in wmap if any(c in k for c in cands)]
        main = [k for k in hits if not k.startswith(("mtp.", "visual.")) and ".mtp." not in k
                and ".visual." not in k and ".vision" not in k]
        hits = main or hits
        return hits[0] if hits else None

    k_in = find(["embed_tokens.weight", "embed_in.weight", "wte.weight", "word_embeddings.weight"])
    if k_in is None:
        raise RuntimeError(f"{name}: no input embedding table found in the checkpoint")
    k_out = find(["lm_head.weight", "embed_out.weight"])
    k_norm = next((k for k in wmap if k.endswith(("model.norm.weight", "final_layer_norm.weight", "ln_f.weight",
                                                  "final_layernorm.weight", "transformer.norm.weight"))
                   and "layers." not in k), None)

    def get(k: str) -> torch.Tensor:
        with safe_open(hf_hub_download(name, wmap[k]), "pt") as f:
            return f.get_tensor(k).float()

    e_in = get(k_in)
    tied = k_out is None or bool(getattr(cfg, "tie_word_embeddings", False))
    e_out = e_in.clone() if k_out is None else get(k_out)
    gain = get(k_norm).flatten()[: e_in.shape[1]] if k_norm else torch.ones(e_in.shape[1])
    logger.info("embeddings for %s: %s rows x %d, tied=%s", name, e_in.shape[0], e_in.shape[1], tied)
    return Embeddings(e_in=e_in, e_out=e_out * gain.unsqueeze(0), tied=tied)


def _embeddings_from_model(name: str, cfg: Any, trust_remote_code: bool) -> Embeddings:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    e_in = model.get_input_embeddings().weight.detach().float()
    e_out = model.get_output_embeddings().weight.detach().float()
    gain = torch.ones(e_in.shape[1])
    for path in ("model.norm", "gpt_neox.final_layer_norm", "transformer.ln_f", "model.final_layernorm"):
        o: Any = model
        ok = True
        for p in path.split("."):
            if not hasattr(o, p):
                ok = False
                break
            o = getattr(o, p)
        if ok and getattr(o, "weight", None) is not None:
            gain = o.weight.detach().float().flatten()[: e_in.shape[1]]
            break
    return Embeddings(e_in=e_in, e_out=e_out * gain.unsqueeze(0), tied=bool(getattr(cfg, "tie_word_embeddings", False)))


__all__ = ["DType", "LoadedModel", "dequantize_fp8_embeddings", "describe_model", "load_embeddings", "load_model",
           "patch_qwen4_exp_indexer"]
