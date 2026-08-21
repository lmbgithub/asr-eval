from asr_eval.dataset import Utterance
from asr_eval.metrics import bootstrap_ci, score_corpus, score_utterance
from asr_eval.normalize import NormalizerConfig


def utt(uid, ref, hyp):
    return Utterance(utterance_id=uid, reference=ref, hypothesis=hyp)


def test_perfect_transcript_scores_zero():
    score = score_utterance(utt("u1", "the cat sat", "the cat sat"))
    assert score.wer == 0.0
    assert score.cer == 0.0


def test_one_substitution_in_three_words():
    score = score_utterance(utt("u1", "the cat sat", "the hat sat"))
    assert score.wer == 1 / 3


def test_punctuation_and_case_do_not_count_as_errors():
    score = score_utterance(utt("u1", "The cat, sat.", "the cat sat"))
    assert score.wer == 0.0


def test_corpus_wer_aggregates_before_dividing():
    # 1 error in 10 words + 0 errors in 2 words == 1/12, not the mean of the
    # per-utterance rates (which would be 1/20).
    long_ref = "a b c d e f g h i j"
    corpus = score_corpus([
        utt("u1", long_ref, "a b c d e f g h i X"),
        utt("u2", "k l", "k l"),
    ])
    assert corpus.wer == 1 / 12
    assert corpus.count == 2


def test_empty_reference_with_empty_hypothesis_is_not_an_error():
    assert score_utterance(utt("u1", "", "")).wer == 0.0


def test_empty_reference_with_output_is_fully_wrong():
    assert score_utterance(utt("u1", "", "hello")).wer == 1.0


def test_normalizer_config_is_recorded_on_the_result():
    cfg = NormalizerConfig(drop_fillers=True)
    corpus = score_corpus([utt("u1", "um yes", "yes")], cfg)
    assert corpus.normalizer == cfg
    assert corpus.wer == 0.0  # the filler is dropped from both sides


def test_worst_ranks_by_error_count_not_rate():
    corpus = score_corpus([
        utt("short", "a", "b"),                    # 1 error, WER 1.00
        utt("long", "a b c d e f", "x y z d e f"), # 3 errors, WER 0.50
    ])
    assert [u.utterance_id for u in corpus.worst(2)] == ["long", "short"]


def test_worst_respects_n():
    corpus = score_corpus([utt(f"u{i}", "a b", "x b") for i in range(5)])
    assert len(corpus.worst(3)) == 3


def test_to_dict_can_omit_utterances():
    corpus = score_corpus([utt("u1", "a", "a")])
    assert "utterances" not in corpus.to_dict(include_utterances=False)
    assert "utterances" in corpus.to_dict()


def test_bootstrap_interval_brackets_the_point_estimate():
    corpus = score_corpus([
        utt(f"u{i}", "a b c d", "a b c d" if i % 4 else "a b c X")
        for i in range(40)
    ])
    interval = bootstrap_ci(corpus.utterances, resamples=300, seed=7)
    assert interval.low <= corpus.wer <= interval.high
    assert interval.confidence == 0.95


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    corpus = score_corpus([utt(f"u{i}", "a b", "a X" if i % 3 else "a b") for i in range(30)])
    a = bootstrap_ci(corpus.utterances, resamples=200, seed=99)
    b = bootstrap_ci(corpus.utterances, resamples=200, seed=99)
    assert (a.low, a.high) == (b.low, b.high)


def test_bootstrap_on_a_perfect_corpus_is_a_point_at_zero():
    corpus = score_corpus([utt(f"u{i}", "a b", "a b") for i in range(10)])
    interval = bootstrap_ci(corpus.utterances, resamples=100, seed=1)
    assert interval.low == 0.0 and interval.high == 0.0


def test_bootstrap_handles_empty_input():
    interval = bootstrap_ci([], resamples=10)
    assert (interval.low, interval.high) == (0.0, 0.0)
