"""Human-readable rendering of scores and comparisons."""

from __future__ import annotations

from asr_eval.align import align, format_alignment
from asr_eval.compare import ComparisonResult
from asr_eval.metrics import CorpusScore, Interval
from asr_eval.normalize import Normalizer


def render_score(
    score: CorpusScore,
    *,
    interval: Interval | None = None,
    worst: int = 5,
    show_alignment: bool = False,
) -> str:
    """Render a corpus score as a plain-text report."""
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("ASR EVALUATION")
    lines.append("=" * 62)
    lines.append(f"utterances       {score.count}")
    lines.append(f"reference words  {score.word.reference_length}")

    wer_line = f"WER              {score.wer:.4f}"
    if interval is not None:
        wer_line += (
            f"  [{interval.low:.4f}, {interval.high:.4f}] "
            f"at {interval.confidence:.0%}"
        )
    lines.append(wer_line)
    lines.append(f"CER              {score.cer:.4f}")
    lines.append("")
    lines.append(
        f"substitutions {score.word.substitutions}   "
        f"deletions {score.word.deletions}   "
        f"insertions {score.word.insertions}"
    )

    # Error-type mix is the first thing to look at when WER moves: the three
    # error classes usually point at three different root causes.
    total = score.word.errors
    if total:
        lines.append(
            f"error mix     S {score.word.substitutions / total:.0%}   "
            f"D {score.word.deletions / total:.0%}   "
            f"I {score.word.insertions / total:.0%}"
        )

    if worst > 0 and score.count:
        lines.append("")
        lines.append(f"--- worst {min(worst, score.count)} utterances by error count ---")
        normalizer = Normalizer(score.normalizer)
        for utt in score.worst(worst):
            lines.append(
                f"[{utt.utterance_id}] WER {utt.wer:.4f}  "
                f"({utt.word.errors} err / {utt.word.reference_length} ref words)"
            )
            lines.append(f"  REF: {utt.reference}")
            lines.append(f"  HYP: {utt.hypothesis}")
            if show_alignment:
                steps = align(normalizer.tokenize(utt.reference), normalizer.tokenize(utt.hypothesis))
                for row in format_alignment(steps).splitlines():
                    lines.append("  " + row)
            lines.append("")

    return "\n".join(lines).rstrip()


def render_comparison(result: ComparisonResult) -> str:
    """Render a baseline comparison as a plain-text report."""
    verdict = "PASS" if result.passed else "FAIL"
    direction = "regression" if result.delta > 0 else "improvement"

    lines: list[str] = []
    lines.append("=" * 62)
    lines.append(f"ASR REGRESSION GATE: {verdict}")
    lines.append("=" * 62)
    lines.append(f"compared utterances  {result.compared_utterances}")
    lines.append(f"baseline WER         {result.baseline_wer:.4f}")
    lines.append(f"candidate WER        {result.candidate_wer:.4f}")
    lines.append(
        f"delta                {result.delta:+.4f} "
        f"({result.relative_delta:+.2%} {direction})"
    )
    if result.p_value is not None:
        lines.append(f"paired bootstrap p   {result.p_value:.4f}")
    if result.reasons:
        lines.append("")
        lines.append("notes:")
        for reason in result.reasons:
            lines.append(f"  - {reason}")
    return "\n".join(lines)
