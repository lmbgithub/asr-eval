import pytest

from asr_eval.compare import GateConfig, compare, paired_bootstrap_p_value
from asr_eval.dataset import Utterance
from asr_eval.metrics import score_corpus
from asr_eval.normalize import NormalizerConfig


def utt(uid, ref, hyp):
    return Utterance(utterance_id=uid, reference=ref, hypothesis=hyp)


def corpus(pairs, cfg=None):
    return score_corpus([utt(uid, r, h) for uid, r, h in pairs], cfg)


def test_identical_runs_pass_with_zero_delta():
    data = [(f"u{i}", "a b c", "a b c") for i in range(10)]
    result = compare(corpus(data), corpus(data))
    assert result.passed
    assert result.delta == 0.0
    assert result.p_value is None  # no regression, so no test is run


def test_improvement_passes():
    base = corpus([(f"u{i}", "a b c d", "a b X d") for i in range(20)])
    cand = corpus([(f"u{i}", "a b c d", "a b c d") for i in range(20)])
    result = compare(base, cand)
    assert result.passed
    assert result.delta < 0
    assert not result.regressed


def test_large_consistent_regression_is_blocked():
    base = corpus([(f"u{i}", "a b c d", "a b c d") for i in range(40)])
    cand = corpus([(f"u{i}", "a b c d", "X Y c d") for i in range(40)])
    result = compare(base, cand)
    assert not result.passed
    assert result.delta > 0
    assert result.p_value is not None and result.p_value < 0.05


def test_tiny_regression_below_tolerance_passes():
    base = corpus([(f"u{i}", "a b c d e f g h i j", "a b c d e f g h i j") for i in range(50)])
    # a single word wrong in one utterance out of 50 -> well under 0.5 points
    data = [(f"u{i}", "a b c d e f g h i j", "a b c d e f g h i j") for i in range(50)]
    data[0] = ("u0", "a b c d e f g h i j", "X b c d e f g h i j")
    result = compare(base, corpus(data), GateConfig(max_absolute_regression=0.01))
    assert result.passed


def test_significance_requirement_can_be_disabled():
    base = corpus([(f"u{i}", "a b c d", "a b c d") for i in range(4)])
    data = [(f"u{i}", "a b c d", "a b c d") for i in range(4)]
    data[0] = ("u0", "a b c d", "X Y Z W")
    cand = corpus(data)
    strict = compare(base, cand, GateConfig(max_absolute_regression=0.0, require_significance=False))
    assert not strict.passed


def test_relative_tolerance_is_applied():
    base = corpus([(f"u{i}", "a b c d", "a b X d") for i in range(40)])
    cand = corpus([(f"u{i}", "a b c d", "a Y X d") for i in range(40)])
    result = compare(
        base,
        cand,
        GateConfig(max_absolute_regression=1.0, max_relative_regression=0.01),
    )
    assert not result.passed
    assert any("relative" in r for r in result.reasons)


def test_mismatched_utterance_sets_are_reported_and_scored_on_the_overlap():
    base = corpus([("u1", "a b", "a b"), ("u2", "a b", "a b")])
    cand = corpus([("u2", "a b", "a b"), ("u3", "a b", "a b")])
    result = compare(base, cand)
    assert result.compared_utterances == 1
    assert result.baseline_only == ("u1",)
    assert result.candidate_only == ("u3",)
    assert any("missing from the candidate" in r for r in result.reasons)


def test_differing_normalizers_are_flagged_as_incomparable():
    data = [("u1", "a b", "a b")]
    base = corpus(data, NormalizerConfig())
    cand = corpus(data, NormalizerConfig(drop_fillers=True))
    result = compare(base, cand)
    assert any("normalizer" in r.lower() for r in result.reasons)


def test_paired_bootstrap_rejects_unequal_lengths():
    base = corpus([("u1", "a", "a")])
    cand = corpus([("u1", "a", "a"), ("u2", "a", "a")])
    with pytest.raises(ValueError):
        paired_bootstrap_p_value(base.utterances, cand.utterances)


def test_paired_bootstrap_is_never_exactly_zero():
    base = corpus([(f"u{i}", "a b c d", "a b c d") for i in range(30)])
    cand = corpus([(f"u{i}", "a b c d", "X Y Z W") for i in range(30)])
    p = paired_bootstrap_p_value(base.utterances, cand.utterances, resamples=200)
    assert 0.0 < p < 0.05


def test_paired_bootstrap_on_empty_input_is_one():
    assert paired_bootstrap_p_value([], []) == 1.0


def test_result_serializes():
    base = corpus([("u1", "a b", "a b")])
    payload = compare(base, base).to_dict()
    assert payload["passed"] is True
    assert "baseline_wer" in payload and "reasons" in payload
