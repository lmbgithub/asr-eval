import pytest

from asr_eval.align import EditOp, align, edit_counts, format_alignment


def test_identical_sequences_have_no_errors():
    counts = edit_counts(["a", "b", "c"], ["a", "b", "c"])
    assert counts.errors == 0
    assert counts.equal == 3
    assert counts.reference_length == 3


def test_substitution_is_counted_once():
    counts = edit_counts(["a", "b", "c"], ["a", "x", "c"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)


def test_deletion_is_a_missing_hypothesis_token():
    counts = edit_counts(["a", "b", "c"], ["a", "c"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)


def test_insertion_is_an_extra_hypothesis_token():
    counts = edit_counts(["a", "c"], ["a", "b", "c"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)


def test_empty_hypothesis_deletes_everything():
    counts = edit_counts(["a", "b"], [])
    assert counts.deletions == 2
    assert counts.errors == 2


def test_empty_reference_inserts_everything():
    counts = edit_counts([], ["a", "b"])
    assert counts.insertions == 2
    assert counts.reference_length == 0


def test_both_empty():
    counts = edit_counts([], [])
    assert counts.errors == 0
    assert counts.reference_length == 0


def test_alignment_path_covers_both_sequences():
    steps = align(["the", "cat", "sat"], ["the", "hat", "sat", "down"])
    ref_tokens = [s.ref for s in steps if s.ref is not None]
    hyp_tokens = [s.hyp for s in steps if s.hyp is not None]
    assert ref_tokens == ["the", "cat", "sat"]
    assert hyp_tokens == ["the", "hat", "sat", "down"]
    assert [s.op for s in steps] == [
        EditOp.EQUAL,
        EditOp.SUBSTITUTE,
        EditOp.EQUAL,
        EditOp.INSERT,
    ]


def test_alignment_is_deterministic_under_ties():
    a, b = ["a", "b"], ["c", "d"]
    first = [s.op for s in align(a, b)]
    for _ in range(5):
        assert [s.op for s in align(a, b)] == first


def test_edit_counts_are_addable():
    total = edit_counts(["a"], ["b"]) + edit_counts(["c"], ["c"])
    assert total.substitutions == 1
    assert total.equal == 1
    assert total.reference_length == 2


def test_counts_indices_point_at_source_tokens():
    steps = align(["a", "b"], ["a", "x"])
    sub = [s for s in steps if s.op is EditOp.SUBSTITUTE][0]
    assert sub.ref_index == 1
    assert sub.hyp_index == 1


def test_format_alignment_renders_three_rows():
    out = format_alignment(align(["the", "cat"], ["the", "hat"]))
    assert "REF:" in out and "HYP:" in out and "OP :" in out
    assert "S" in out


def test_format_alignment_handles_empty():
    assert format_alignment([]) == ""


@pytest.mark.parametrize(
    "ref,hyp,expected",
    [
        (["a"], ["a"], 0),
        (["a"], ["b"], 1),
        (["a", "b", "c"], ["a"], 2),
        ([], ["a", "b", "c"], 3),
    ],
)
def test_error_totals(ref, hyp, expected):
    assert edit_counts(ref, hyp).errors == expected
