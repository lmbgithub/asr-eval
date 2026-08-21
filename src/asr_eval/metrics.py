"""WER, CER, and the uncertainty around them.

A corpus WER reported without an interval invites the most common mistake in
model evaluation: treating a 0.3-point move on a small set as a regression. The
bootstrap here exists so a gate can ask "is this move larger than the noise on
this set?" instead of "is this move larger than zero?".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from asr_eval.align import EditCounts, edit_counts
from asr_eval.dataset import Utterance
from asr_eval.normalize import Normalizer, NormalizerConfig


@dataclass(frozen=True)
class UtteranceScore:
    """Per-utterance word- and character-level result."""

    utterance_id: str
    word: EditCounts
    char: EditCounts
    reference: str
    hypothesis: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def wer(self) -> float:
        return _ratio(self.word.errors, self.word.reference_length)

    @property
    def cer(self) -> float:
        return _ratio(self.char.errors, self.char.reference_length)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.utterance_id,
            "wer": round(self.wer, 6),
            "cer": round(self.cer, 6),
            "word": self.word.to_dict(),
            "char": self.char.to_dict(),
        }


@dataclass(frozen=True)
class Interval:
    """A bootstrap confidence interval."""

    low: float
    high: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CorpusScore:
    """Corpus-level aggregate over a set of utterance scores."""

    utterances: tuple[UtteranceScore, ...]
    word: EditCounts
    char: EditCounts
    normalizer: NormalizerConfig

    @property
    def wer(self) -> float:
        return _ratio(self.word.errors, self.word.reference_length)

    @property
    def cer(self) -> float:
        return _ratio(self.char.errors, self.char.reference_length)

    @property
    def count(self) -> int:
        return len(self.utterances)

    def worst(self, n: int = 10) -> list[UtteranceScore]:
        """Return the `n` utterances with the highest WER, ties broken by id.

        Sorting by error *rate* alone surfaces three-word utterances with one
        error. Ranking by error count first keeps the list actionable.
        """
        return sorted(
            self.utterances,
            key=lambda u: (-u.word.errors, -u.wer, u.utterance_id),
        )[:n]

    def to_dict(self, *, include_utterances: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "corpus": {
                "wer": round(self.wer, 6),
                "cer": round(self.cer, 6),
                "utterances": self.count,
                "word": self.word.to_dict(),
                "char": self.char.to_dict(),
            },
            "normalizer": self.normalizer.to_dict(),
        }
        if include_utterances:
            data["utterances"] = [u.to_dict() for u in self.utterances]
        return data


def _ratio(numerator: int, denominator: int) -> float:
    """Errors over reference length, with an explicit empty-reference rule.

    An empty reference has no denominator. Returning 0.0 for an empty hypothesis
    (nothing expected, nothing produced) and 1.0 otherwise (everything produced
    is an insertion) keeps the corpus aggregate finite without silently dropping
    the utterance from the count.
    """
    if denominator == 0:
        return 0.0 if numerator == 0 else 1.0
    return numerator / denominator


def score_utterance(
    utterance: Utterance,
    normalizer: Normalizer | None = None,
) -> UtteranceScore:
    """Score one utterance at word and character level."""
    norm = normalizer or Normalizer()
    ref_words = norm.tokenize(utterance.reference)
    hyp_words = norm.tokenize(utterance.hypothesis)
    ref_chars = norm.characters(utterance.reference)
    hyp_chars = norm.characters(utterance.hypothesis)
    return UtteranceScore(
        utterance_id=utterance.utterance_id,
        word=edit_counts(ref_words, hyp_words),
        char=edit_counts(ref_chars, hyp_chars),
        reference=utterance.reference,
        hypothesis=utterance.hypothesis,
        metadata=dict(utterance.metadata),
    )


def score_corpus(
    utterances: Sequence[Utterance],
    config: NormalizerConfig | None = None,
) -> CorpusScore:
    """Score a corpus, aggregating errors before dividing.

    Corpus WER is total errors over total reference tokens, not the mean of
    per-utterance WERs. The two differ whenever utterance lengths differ, and
    the unweighted mean lets a handful of very short utterances dominate.
    """
    cfg = config or NormalizerConfig()
    normalizer = Normalizer(cfg)
    scores = [score_utterance(u, normalizer) for u in utterances]

    word_total = EditCounts()
    char_total = EditCounts()
    for score in scores:
        word_total = word_total + score.word
        char_total = char_total + score.char

    return CorpusScore(
        utterances=tuple(scores),
        word=word_total,
        char=char_total,
        normalizer=cfg,
    )


def bootstrap_ci(
    scores: Sequence[UtteranceScore],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 1234,
    level: str = "word",
) -> Interval:
    """Bootstrap a confidence interval for corpus WER (or CER).

    Resampling is over *utterances*, not tokens: utterances are the independent
    unit here, and resampling tokens would understate the interval by ignoring
    the correlation of errors within an utterance.

    `seed` is fixed by default so a gate is reproducible. A gate whose verdict
    changes between two identical runs cannot be trusted to block a release.
    """
    if not scores:
        return Interval(0.0, 0.0, confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")

    counts = [(s.word if level == "word" else s.char) for s in scores]
    rng = random.Random(seed)
    n = len(counts)

    estimates: list[float] = []
    for _ in range(resamples):
        errors = 0
        length = 0
        for _ in range(n):
            picked = counts[rng.randrange(n)]
            errors += picked.errors
            length += picked.reference_length
        estimates.append(_ratio(errors, length))

    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    low = estimates[_percentile_index(len(estimates), tail)]
    high = estimates[_percentile_index(len(estimates), 1.0 - tail)]
    return Interval(low=low, high=high, confidence=confidence)


def _percentile_index(n: int, q: float) -> int:
    """Index into a sorted list of length `n` for quantile `q`, clamped."""
    idx = int(round(q * (n - 1)))
    return max(0, min(n - 1, idx))
