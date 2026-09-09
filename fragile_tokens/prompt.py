"""The copy prompt: ``head + pre_text + <text ids> + pre_copy + <copy ids>``, in raw or chat framing.

Every score the package produces reads log-probabilities inside the copy span of a sequence built
this way. The head carries the instruction and the few-shot demonstrations; ``pre_text`` and
``pre_copy`` carry the ``Text:`` and ``Copy:`` markers (and, in chat framing, the template's turn
boundaries). The copy span is always last, so a token's copy position is fixed by arithmetic and no
padding is ever needed within a context.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from .config import DEFAULT_DEMOS, Framing
from .types import TokenizerLike, encode

logger = logging.getLogger(__name__)

INSTRUCTION = "Repeat the text exactly.\n"
_TEXT_SENTINEL = "\x01"
_COPY_SENTINEL = "\x02"


@dataclass(frozen=True)
class CopyPrompt:
    """Token-id parts of the copy prompt for one tokenizer and one framing."""

    head: tuple[int, ...]
    pre_text: tuple[int, ...]
    pre_copy: tuple[int, ...]
    framing: Framing
    demos: tuple[str, ...] = DEFAULT_DEMOS

    @classmethod
    def build(cls, tok: TokenizerLike, framing: Framing = Framing.AUTO,
              demos: Sequence[str] = DEFAULT_DEMOS) -> "CopyPrompt":
        """Build the prompt parts. ``Framing.AUTO`` picks chat when the tokenizer has a usable chat
        template and raw otherwise; a chat template that cannot be split around the copy span falls
        back to raw with a warning."""
        demos_t = tuple(demos)
        raw_head = encode(tok, INSTRUCTION)
        for d in demos_t:
            raw_head += encode(tok, "Text:") + encode(tok, d) + encode(tok, "\nCopy:") + encode(tok, d) + encode(tok, "\n")
        raw = cls(tuple(raw_head), tuple(encode(tok, "Text:")), tuple(encode(tok, "\nCopy:")), Framing.RAW, demos_t)
        if framing is Framing.RAW:
            return raw
        chat = cls._chat_parts(tok, demos_t)
        if chat is None:
            if framing is Framing.CHAT:
                logger.warning("chat framing requested but the tokenizer's chat template could not be used; using raw")
            return raw
        return cls(chat[0], chat[1], chat[2], Framing.CHAT, demos_t)

    @staticmethod
    def _chat_parts(tok: TokenizerLike, demos: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
        apply = getattr(tok, "apply_chat_template", None)
        if apply is None or getattr(tok, "chat_template", None) is None:
            return None
        user = INSTRUCTION + "".join(f"Text:{d}\nCopy:{d}\n" for d in demos) + "Text:" + _TEXT_SENTINEL
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": "Copy:" + _COPY_SENTINEL}]
        rendered: Any = None
        for kw in ({"enable_thinking": False}, {}):
            try:
                rendered = apply(msgs, tokenize=False, continue_final_message=True, **kw)
                break
            except Exception as exc:  # templates differ in the switches they accept
                logger.debug("chat template rejected %s: %s", kw, exc)
        if not isinstance(rendered, str) or _TEXT_SENTINEL not in rendered or _COPY_SENTINEL not in rendered:
            return None
        before_text, rest = rendered.split(_TEXT_SENTINEL, 1)
        before_copy, after_copy = rest.split(_COPY_SENTINEL, 1)
        if after_copy.strip():
            logger.debug("chat template appended %r after the copy span; it is dropped so the copy stays last", after_copy)
        return tuple(encode(tok, before_text)), (), tuple(encode(tok, before_copy))

    # ---- assembly ------------------------------------------------------------------------------
    @property
    def text_start(self) -> int:
        return len(self.head) + len(self.pre_text)

    def copy_start(self, n_text: int) -> int:
        return self.text_start + n_text + len(self.pre_copy)

    def sequence(self, text_ids: Sequence[int]) -> list[int]:
        """The full prompt with ``text_ids`` as both the text and the copy span."""
        ids = [int(t) for t in text_ids]
        return list(self.head) + list(self.pre_text) + ids + list(self.pre_copy) + ids

    def single(self, token_id: int) -> list[int]:
        """The single-token probe: the token alone as text and copy."""
        return self.sequence([token_id])

    def positions(self, n_text: int, slot: int) -> tuple[int, int]:
        """(text position, copy position) of the slot in a sequence with ``n_text`` text tokens."""
        if not 0 <= slot < n_text:
            raise ValueError(f"slot {slot} outside a text of {n_text} tokens")
        return self.text_start + slot, self.copy_start(n_text) + slot

    def describe(self, tok: TokenizerLike) -> dict[str, Any]:
        return {
            "framing": self.framing.value,
            "demos": list(self.demos),
            "head_text": tok.decode(list(self.head)),
            "pre_text": tok.decode(list(self.pre_text)),
            "pre_copy": tok.decode(list(self.pre_copy)),
            "head_tokens": len(self.head),
        }


__all__ = ["INSTRUCTION", "CopyPrompt"]
