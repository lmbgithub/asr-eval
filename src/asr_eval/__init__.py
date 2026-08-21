"""Evaluation and regression-gating toolkit for speech recognition systems."""

from asr_eval.align import EditOp, align, edit_counts
from asr_eval.dataset import Utterance, load_manifest
from asr_eval.metrics import (
    CorpusScore,
    UtteranceScore,
    bootstrap_ci,
    score_corpus,
    score_utterance,
)
from asr_eval.normalize import Normalizer, NormalizerConfig

__version__ = "0.1.0"

__all__ = [
    "EditOp",
    "align",
    "edit_counts",
    "Utterance",
    "load_manifest",
    "CorpusScore",
    "UtteranceScore",
    "bootstrap_ci",
    "score_corpus",
    "score_utterance",
    "Normalizer",
    "NormalizerConfig",
    "__version__",
]
