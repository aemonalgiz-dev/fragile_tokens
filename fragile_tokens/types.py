"""Structural types for the model and tokenizer, so the library and its tests do not depend on the
concrete transformers classes."""
from __future__ import annotations

from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@runtime_checkable
class TokenizerLike(Protocol):
    """The slice of a Hugging Face tokenizer this package uses."""

    pad_token_id: int | None
    eos_token_id: int | None

    def __call__(self, text: str, add_special_tokens: bool = ...) -> Mapping[str, Any]: ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...

    def __len__(self) -> int: ...


class ModelOutputLike(Protocol):
    logits: torch.Tensor


@runtime_checkable
class CausalLMLike(Protocol):
    """A causal language model: ``model(input_ids=...)`` returns an object with ``.logits``."""

    def __call__(self, *, input_ids: torch.Tensor, **kwargs: Any) -> ModelOutputLike: ...

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...


def encode(tok: TokenizerLike, text: str) -> list[int]:
    """Token ids of ``text`` without special tokens."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [int(i) for i in ids]


__all__ = ["BoolArray", "CausalLMLike", "FloatArray", "IntArray", "ModelOutputLike", "TokenizerLike", "encode"]
