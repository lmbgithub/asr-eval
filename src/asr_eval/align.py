"""Levenshtein alignment with a full backtrace.

Error *counts* alone tell you a system got worse. The alignment tells you how:
a spike in deletions is usually a truncation or endpointing bug, a spike in
insertions on a clean reference is a hallucination signature, and a spike in
substitutions is typically vocabulary or accent drift. Those are different
on-call pages, so the alignment is kept rather than discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class EditOp(str, Enum):
    """A single alignment operation between reference and hypothesis."""

    EQUAL = "equal"
    SUBSTITUTE = "substitute"
    DELETE = "delete"  # present in reference, missing from hypothesis
    INSERT = "insert"  # present in hypothesis, absent from reference


@dataclass(frozen=True)
class AlignmentStep:
    """One aligned position. `ref`/`hyp` are None where the side has no token."""

    op: EditOp
    ref: str | None
    hyp: str | None
    ref_index: int | None
    hyp_index: int | None


@dataclass(frozen=True)
class EditCounts:
    """Aggregate edit-operation counts for one alignment."""

    equal: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        return self.equal + self.substitutions + self.deletions

    @property
    def hypothesis_length(self) -> int:
        return self.equal + self.substitutions + self.insertions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        if not isinstance(other, EditCounts):
            return NotImplemented
        return EditCounts(
            equal=self.equal + other.equal,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "equal": self.equal,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "reference_length": self.reference_length,
        }


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> list[AlignmentStep]:
    """Align two token sequences and return the full edit path.

    Uses the standard Levenshtein dynamic-programming table with unit costs and
    a backtrace. Runs in O(len(reference) * len(hypothesis)) time and memory,
    which is fine at utterance granularity; the corpus loop stays linear in the
    number of utterances because each alignment is independent.

    Ties are resolved toward substitution, then deletion, then insertion, so the
    output is deterministic across runs and platforms. Determinism matters here:
    a regression gate that reports different counts on identical input is worse
    than no gate at all.
    """
    n, m = len(reference), len(hypothesis)

    # cost[i][j] = edit distance between reference[:i] and hypothesis[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j

    for i in range(1, n + 1):
        ref_tok = reference[i - 1]
        row, prev_row = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            if ref_tok == hypothesis[j - 1]:
                row[j] = prev_row[j - 1]
            else:
                row[j] = 1 + min(
                    prev_row[j - 1],  # substitute
                    prev_row[j],      # delete
                    row[j - 1],       # insert
                )

    steps: list[AlignmentStep] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            steps.append(AlignmentStep(EditOp.EQUAL, reference[i - 1], hypothesis[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            steps.append(AlignmentStep(EditOp.SUBSTITUTE, reference[i - 1], hypothesis[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            steps.append(AlignmentStep(EditOp.DELETE, reference[i - 1], None, i - 1, None))
            i -= 1
        else:
            steps.append(AlignmentStep(EditOp.INSERT, None, hypothesis[j - 1], None, j - 1))
            j -= 1

    steps.reverse()
    return steps


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Return aggregate edit counts for one reference/hypothesis pair."""
    counts = {EditOp.EQUAL: 0, EditOp.SUBSTITUTE: 0, EditOp.DELETE: 0, EditOp.INSERT: 0}
    for step in align(reference, hypothesis):
        counts[step.op] += 1
    return EditCounts(
        equal=counts[EditOp.EQUAL],
        substitutions=counts[EditOp.SUBSTITUTE],
        deletions=counts[EditOp.DELETE],
        insertions=counts[EditOp.INSERT],
    )


def format_alignment(steps: Sequence[AlignmentStep], width: int = 100) -> str:
    """Render an alignment as three aligned rows (REF / HYP / OP) for humans."""
    if not steps:
        return ""
    ref_cells, hyp_cells, op_cells = [], [], []
    for step in steps:
        ref = step.ref if step.ref is not None else "*"
        hyp = step.hyp if step.hyp is not None else "*"
        marker = {
            EditOp.EQUAL: "",
            EditOp.SUBSTITUTE: "S",
            EditOp.DELETE: "D",
            EditOp.INSERT: "I",
        }[step.op]
        cell_width = max(len(ref), len(hyp), len(marker))
        ref_cells.append(ref.ljust(cell_width))
        hyp_cells.append(hyp.ljust(cell_width))
        op_cells.append(marker.ljust(cell_width))

    lines: list[str] = []
    start = 0
    while start < len(ref_cells):
        end, length = start, 0
        while end < len(ref_cells) and length + len(ref_cells[end]) + 1 <= width:
            length += len(ref_cells[end]) + 1
            end += 1
        end = max(end, start + 1)
        lines.append("REF: " + " ".join(ref_cells[start:end]))
        lines.append("HYP: " + " ".join(hyp_cells[start:end]))
        lines.append("OP : " + " ".join(op_cells[start:end]))
        lines.append("")
        start = end
    return "\n".join(lines).rstrip()
