"""fragile_tokens: find vocabulary entries that pass the single-token glitch probe and fail to copy
inside ordinary text.

The measurement is the one described in *Fragile Tokens* (Gordon, 2026): every token is placed at a
random slot of each context in a bank of random common-word sequences, the model is asked to repeat
the sequence under a fixed few-shot copy prompt, and the teacher-forced log-probability of the token
at its copy position is recorded. A token's fragility is the share of contexts in which that
log-probability falls below a threshold. Tokens that pass the probe alone but are fragile in context
are the ones every isolation-based detector calls healthy.

Public entry points: :class:`FragileTokenDetector` (library), ``fragile-tokens`` (command line).
"""
from __future__ import annotations

from ._version import __version__
from .classify import FailureMode, TokenClass, classify_matrix, glitch_reference
from .config import Framing, ScanConfig
from .contexts import Context, ContextBank, build_bank, common_word_ids
from .detector import FragileTokenDetector
from .geometry import GeometryScores, auc, glitch_proximity, glitch_projection
from .prompt import CopyPrompt
from .report import ScanResult, TokenReport
from .scoring import FragilityMatrix

__all__ = [
    "__version__",
    "Context",
    "ContextBank",
    "CopyPrompt",
    "FailureMode",
    "FragileTokenDetector",
    "FragilityMatrix",
    "Framing",
    "GeometryScores",
    "ScanConfig",
    "ScanResult",
    "TokenClass",
    "TokenReport",
    "auc",
    "build_bank",
    "classify_matrix",
    "common_word_ids",
    "glitch_projection",
    "glitch_proximity",
    "glitch_reference",
]
