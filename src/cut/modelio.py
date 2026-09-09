"""Model / tokenizer loading and batched next-token distribution utilities."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "EleutherAI/pythia-1.4b"


def load(model_name: str = DEFAULT_MODEL, device: str | None = None, dtype=torch.float16):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    kw = {"dtype": dtype if device == "cuda" else torch.float32}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(device)
    except TypeError:  # transformers < 5
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=kw["dtype"]).to(device)
    model.eval()
    return model, tok, device


@torch.no_grad()
def next_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """Logits at the final position only. input_ids: (B, T) left-aligned, equal length."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def seq_logprobs(model, input_ids: torch.Tensor, target_mask: torch.Tensor):
    """Mean log p of positions where target_mask is True (teacher forced).

    input_ids: (B, T). target_mask: (B, T) bool, True on positions to score.
    Returns (B,) mean logprob over scored positions.
    """
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :].float()
    tgt = input_ids[:, 1:]
    mask = target_mask[:, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    lp = lp * mask
    denom = mask.sum(dim=1).clamp(min=1)
    return lp.sum(dim=1) / denom
