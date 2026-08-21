"""Baseline comparison and release gating.

The gate answers one question: given the noise on this evaluation set, is the
new system meaningfully worse than the baseline? Two conditions must both hold
before it blocks, because either one alone produces gates teams learn to ignore:

  * the regression clears an absolute tolerance (ignore trivia), and
  * a paired bootstrap says the move is unlikely to be sampling noise.

A gate that fires on noise gets bypassed within a month, and a bypassed gate
protects nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from asr_eval.metrics import CorpusScore, UtteranceScore, _ratio


@dataclass(frozen=True)
class GateConfig:
    """Thresholds controlling when a comparison fails."""

    max_absolute_regression: float = 0.005  # 0.5 WER points
    max_relative_regression: float | None = None  # e.g. 0.02 for 2% relative
    significance: float = 0.05
    resamples: int = 1000
    seed: int = 1234
    require_significance: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_absolute_regression": self.max_absolute_regression,
            "max_relative_regression": self.max_relative_regression,
            "significance": self.significance,
            "resamples": self.resamples,
            "seed": self.seed,
            "require_significance": self.require_significance,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing a candidate corpus score against a baseline."""

    baseline_wer: float
    candidate_wer: float
    delta: float
    relative_delta: float
    p_value: float | None
    passed: bool
    reasons: tuple[str, ...]
    compared_utterances: int
    baseline_only: tuple[str, ...] = ()
    candidate_only: tuple[str, ...] = ()

    @property
    def regressed(self) -> bool:
        return self.delta > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_wer": round(self.baseline_wer, 6),
            "candidate_wer": round(self.candidate_wer, 6),
            "delta": round(self.delta, 6),
            "relative_delta": round(self.relative_delta, 6),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "compared_utterances": self.compared_utterances,
            "baseline_only": list(self.baseline_only),
            "candidate_only": list(self.candidate_only),
        }


def paired_bootstrap_p_value(
    baseline: Sequence[UtteranceScore],
    candidate: Sequence[UtteranceScore],
    *,
    resamples: int = 1000,
    seed: int = 1234,
) -> float:
    """One-sided p-value that the candidate is no worse than the baseline.

    Paired: each resample draws the same utterance indices from both systems, so
    the comparison is not polluted by which utterances happened to be sampled.
    The returned value is the fraction of resamples in which the candidate is
    not worse; a small value means the regression held up under resampling.

    Uses the (1 + count) / (1 + resamples) correction so the result is never
    exactly zero, which would overstate certainty the resample count cannot
    support.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap requires equal-length, aligned sequences")
    if not baseline:
        return 1.0

    rng = random.Random(seed)
    n = len(baseline)
    not_worse = 0

    for _ in range(resamples):
        b_err = b_len = c_err = c_len = 0
        for _ in range(n):
            idx = rng.randrange(n)
            b, c = baseline[idx].word, candidate[idx].word
            b_err += b.errors
            b_len += b.reference_length
            c_err += c.errors
            c_len += c.reference_length
        if _ratio(c_err, c_len) <= _ratio(b_err, b_len):
            not_worse += 1

    return (1.0 + not_worse) / (1.0 + resamples)


def compare(
    baseline: CorpusScore,
    candidate: CorpusScore,
    config: GateConfig | None = None,
) -> ComparisonResult:
    """Compare two corpus scores and decide whether the candidate may ship."""
    cfg = config or GateConfig()

    base_by_id = {u.utterance_id: u for u in baseline.utterances}
    cand_by_id = {u.utterance_id: u for u in candidate.utterances}
    shared = sorted(set(base_by_id) & set(cand_by_id))
    baseline_only = tuple(sorted(set(base_by_id) - set(cand_by_id)))
    candidate_only = tuple(sorted(set(cand_by_id) - set(base_by_id)))

    reasons: list[str] = []

    # Compare on the shared subset only. Scoring two different sets against each
    # other silently changes the denominator and produces a "regression" that is
    # really a change of test set.
    paired_base = [base_by_id[i] for i in shared]
    paired_cand = [cand_by_id[i] for i in shared]

    base_wer = _corpus_wer(paired_base) if shared else baseline.wer
    cand_wer = _corpus_wer(paired_cand) if shared else candidate.wer

    delta = cand_wer - base_wer
    relative = delta / base_wer if base_wer > 0 else (0.0 if delta == 0 else float("inf"))

    if baseline_only:
        reasons.append(
            f"{len(baseline_only)} utterance(s) in baseline are missing from the candidate run"
        )
    if candidate_only:
        reasons.append(
            f"{len(candidate_only)} utterance(s) in the candidate run are absent from the baseline"
        )
    if baseline.normalizer != candidate.normalizer:
        reasons.append(
            "normalizer configuration differs between runs; WER values are not comparable"
        )

    p_value: float | None = None
    if shared and delta > 0:
        p_value = paired_bootstrap_p_value(
            paired_base, paired_cand, resamples=cfg.resamples, seed=cfg.seed
        )

    exceeds_absolute = delta > cfg.max_absolute_regression
    exceeds_relative = (
        cfg.max_relative_regression is not None and relative > cfg.max_relative_regression
    )
    is_significant = p_value is not None and p_value < cfg.significance

    passed = True
    if exceeds_absolute or exceeds_relative:
        if not cfg.require_significance or is_significant:
            passed = False
            if exceeds_absolute:
                reasons.append(
                    f"WER regressed by {delta:.4f} absolute, above the "
                    f"{cfg.max_absolute_regression:.4f} tolerance"
                )
            if exceeds_relative:
                reasons.append(
                    f"WER regressed by {relative:.2%} relative, above the "
                    f"{cfg.max_relative_regression:.2%} tolerance"
                )
            if p_value is not None:
                reasons.append(f"paired bootstrap p={p_value:.4f} < {cfg.significance}")
        else:
            reasons.append(
                f"WER regressed by {delta:.4f} but the paired bootstrap could not "
                f"separate it from noise (p={p_value:.4f} >= {cfg.significance}); "
                "not blocking"
            )

    # A "normalizer differs" note is advisory on its own, but combined with a
    # regression it is the most likely explanation, so it is surfaced either way
    # rather than being folded into the pass/fail decision.
    return ComparisonResult(
        baseline_wer=base_wer,
        candidate_wer=cand_wer,
        delta=delta,
        relative_delta=relative,
        p_value=p_value,
        passed=passed,
        reasons=tuple(reasons),
        compared_utterances=len(shared),
        baseline_only=baseline_only,
        candidate_only=candidate_only,
    )


def _corpus_wer(scores: Sequence[UtteranceScore]) -> float:
    errors = sum(s.word.errors for s in scores)
    length = sum(s.word.reference_length for s in scores)
    return _ratio(errors, length)
