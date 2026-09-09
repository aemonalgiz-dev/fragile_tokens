"""A fake tokenizer and a fake copying model, so the pipeline is exercised end to end on the CPU
without downloading or running any real model."""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pytest
import torch

# vocabulary: specials, then punctuation/markers, then 200 common ' wNNN' words, then the test tokens
SPECIALS = ["<pad>", "<eos>"]
PIECES = ["Repeat", " the", " text", " exactly", ".", "\n", "Text", ":", "Copy", " apple", " pie", " is", " good",
          " quick", " brown", " fox", "<", "|", "im_start", ">", "user", "assistant", "\n\n", " "]
def common(i: int) -> str:
    """The i-th fake common word: ' waa', ' wab', ... (alphabetic, lowercase, like real BPE word pieces)."""
    return f" w{chr(97 + i // 26)}{chr(97 + i % 26)}"


COMMON = [common(i) for i in range(200)]
TEST_TOKENS = [" according", " zebra", " Председа", "ঈ", " realt", " helper"]
VOCAB = SPECIALS + PIECES + COMMON + TEST_TOKENS
ID = {s: i for i, s in enumerate(VOCAB)}

TOKEN_RE = re.compile(r"\n\n|\n| [A-Za-zА-Яа-яঀ-৿][^\s:.<>|]*| |[A-Za-z_]+|[^\sA-Za-z]")


class FakeTokenizer:
    """Greedy longest-match over a fixed vocabulary; unknown pieces raise so tests stay honest."""

    pad_token_id: int | None = ID["<pad>"]
    eos_token_id: int | None = ID["<eos>"]
    all_special_ids = [ID["<pad>"], ID["<eos>"]]
    chat_template: str | None = None

    def __call__(self, text: str, add_special_tokens: bool = True) -> Mapping[str, Any]:
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            m = TOKEN_RE.match(text, pos)
            if not m:
                raise ValueError(f"cannot tokenize {text[pos:pos + 12]!r}")
            piece = m.group(0)
            if piece not in ID:
                raise ValueError(f"unknown piece {piece!r}")
            ids.append(ID[piece])
            pos = m.end()
        return {"input_ids": ids}

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(VOCAB[int(i)] for i in token_ids)

    def __len__(self) -> int:
        return len(VOCAB)


class ChatFakeTokenizer(FakeTokenizer):
    chat_template = "fake"

    def apply_chat_template(self, msgs: list[dict[str, str]], tokenize: bool = False, continue_final_message: bool = False,
                            **kw: Any) -> str:
        if "enable_thinking" in kw:
            raise TypeError("unexpected keyword")
        return "<|im_start|>user\n" + msgs[0]["content"] + "\n<|im_start|>assistant\n" + msgs[1]["content"]


class FakeCopyModel:
    """Predicts the copy of the text span. ``glitch_ids`` are deleted (the model emits the next context
    word) in every context and fail alone; ``fragile_ids`` are deleted whenever the preceding context id
    is even, so their fragility is about one half; everything else copies perfectly. The logit gap sets
    the confidence: ``sharp`` for near-zero entropy, otherwise a softer distribution."""

    def __init__(self, vocab_size: int, glitch_ids: Sequence[int] = (), fragile_ids: Sequence[int] = (),
                 sharp: bool = True) -> None:
        self.v = vocab_size
        self.glitch = set(glitch_ids)
        self.fragile = set(fragile_ids)
        self.gap = 12.0 if sharp else 1.2
        self.calls = 0
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=4, num_hidden_layers=1, tie_word_embeddings=True)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._param

    def eval(self) -> "FakeCopyModel":
        return self

    def _spans(self, ids: list[int]) -> tuple[int, int, int] | None:
        text_marker, copy_marker = [ID["Text"], ID[":"]], [ID["\n"], ID["Copy"], ID[":"]]
        cs = None
        for j in range(len(ids) - 2, -1, -1):
            if ids[j: j + 3] == copy_marker:
                cs = j + 3
                break
        if cs is None:
            return None
        ts = None
        for j in range(cs - 4, -1, -1):
            if ids[j: j + 2] == text_marker:
                ts = j + 2
                break
        if ts is None:
            return None
        return ts, cs - 3, cs  # text start, text end (exclusive), copy start

    def __call__(self, *, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        self.calls += 1
        b, n = input_ids.shape
        logits = torch.zeros(b, n, self.v)
        for r in range(b):
            ids = input_ids[r].tolist()
            spans = self._spans(ids)
            if spans is None:
                continue
            ts, te, cs = spans
            text = ids[ts:te]
            for j in range(cs - 1, n):  # position j predicts ids[j + 1]
                k = j + 1 - cs
                if k >= len(text):
                    break
                target = text[k]
                prev = text[k - 1] if k > 0 else None
                emit = target
                if target in self.glitch or (target in self.fragile and prev is not None and prev % 2 == 0):
                    emit = text[k + 1] if k + 1 < len(text) else ID["\n"]
                if len(text) == 1 and target in self.glitch:
                    emit = ID["\n"]
                logits[r, j, emit] = self.gap
        return SimpleNamespace(logits=logits)


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def chat_tok() -> ChatFakeTokenizer:
    return ChatFakeTokenizer()


@pytest.fixture
def glitch_ids() -> list[int]:
    return [ID[" Председа"], ID["ঈ"]]


@pytest.fixture
def fragile_ids() -> list[int]:
    return [ID[" according"], ID[" realt"]]


@pytest.fixture
def model(glitch_ids: list[int], fragile_ids: list[int]) -> FakeCopyModel:
    return FakeCopyModel(len(VOCAB), glitch_ids, fragile_ids)


@pytest.fixture
def ids() -> dict[str, int]:
    return ID


def make_arrays(n: int, c: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, c)).astype(np.float32), rng.integers(0, 10, size=(n, c)).astype(np.int64)
