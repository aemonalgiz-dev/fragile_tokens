"""One place that decides how the copy prompt is framed.

Every copy score in the pipeline is built as  head + pre_t + <context ids> + pre_c + <copy ids>
and reads log-probabilities inside the copy span. Two framings:

  raw   (default; every run before 7 September 2026)
        head  = "Repeat the text exactly.\\n" + few-shot demos     pre_t = "Text:"     pre_c = "\\nCopy:"
        Plain text, no chat template. Works on OLMo-2, Qwen2.5, Qwen3.

  chat  (GLITCH_PROMPT_STYLE=chat)
        The same instruction, demos and "Text:" inside the USER turn of the model's chat template,
        and "Copy:" at the start of the ASSISTANT turn, with thinking disabled where the template
        knows the switch. Needed for Qwen3.5 / Qwen3.8: under the raw framing those models put
        0.87 on a blank line after "Copy:" and then copy the text at the start of a new line, so
        the copied token loses its leading space and no exact-id copy can succeed (Qwen3.8-27B,
        filler pool 3 of ~2,000 common words). Inside an assistant turn they copy normally.

The choice is recorded in every fragility JSON as "prompt_style". Within a model every
measurement uses one framing, so fragility remains a within-model quantity; across models the
paper reports the framing alongside the gate.
"""
from __future__ import annotations
import os

STYLE = os.environ.get("GLITCH_PROMPT_STYLE", "raw")


def copy_prompt_parts(tok, raw_head, demos):
    """Return (head, pre_t, pre_c) as id lists for the active framing. `raw_head` is the
    caller's already-built raw head (used unchanged in raw style); `demos` the few-shot strings."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    if STYLE != "chat":
        return raw_head, e("Text:"), e("\nCopy:")
    user = "Repeat the text exactly.\n" + "".join(f"Text:{d}\nCopy:{d}\n" for d in demos) + "Text:\x01"
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": "Copy:\x02"}]
    s = None
    for kw in ({"enable_thinking": False}, {}):
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True, **kw)
            break
        except Exception:
            continue
    if s is None or "\x01" not in s or "\x02" not in s:
        return raw_head, e("Text:"), e("\nCopy:")
    a, rest = s.split("\x01", 1)
    b, c = rest.split("\x02", 1)
    if c.strip():                       # the template appended something after the copy; keep it out
        b = b  # nothing to do: the copy span must be last, so we drop `c` deliberately
    return e(a), [], e(b)
