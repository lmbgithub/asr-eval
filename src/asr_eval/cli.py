"""Command-line interface: `asr-eval run` and `asr-eval compare`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from asr_eval import __version__
from asr_eval.compare import GateConfig, compare
from asr_eval.dataset import ManifestError, load_manifest
from asr_eval.metrics import (
    CorpusScore,
    UtteranceScore,
    bootstrap_ci,
    score_corpus,
)
from asr_eval.align import EditCounts
from asr_eval.normalize import NormalizerConfig
from asr_eval.report import render_comparison, render_score

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr-eval",
        description="Evaluate ASR output and gate releases on WER regressions.",
    )
    parser.add_argument("--version", action="version", version=f"asr-eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score a manifest of reference/hypothesis pairs")
    run.add_argument("manifest", help="path to a JSON Lines manifest")
    run.add_argument("--json", dest="json_out", help="write the full result as JSON to this path")
    run.add_argument("--worst", type=int, default=5, help="how many worst utterances to print")
    run.add_argument("--alignment", action="store_true", help="print token alignments for worst utterances")
    run.add_argument("--no-ci", action="store_true", help="skip the bootstrap confidence interval")
    run.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples")
    run.add_argument("--seed", type=int, default=1234, help="bootstrap seed (fixed for reproducibility)")
    run.add_argument("--quiet", action="store_true", help="suppress the human-readable report")
    _add_normalizer_flags(run)

    cmp_ = sub.add_parser("compare", help="compare a run against a stored baseline")
    cmp_.add_argument("baseline", help="baseline JSON produced by `asr-eval run --json`")
    cmp_.add_argument("candidate", help="candidate manifest, or candidate JSON from `run --json`")
    cmp_.add_argument("--max-absolute", type=float, default=0.005, help="absolute WER regression tolerance")
    cmp_.add_argument("--max-relative", type=float, default=None, help="relative WER regression tolerance")
    cmp_.add_argument("--significance", type=float, default=0.05, help="paired-bootstrap alpha")
    cmp_.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples")
    cmp_.add_argument("--seed", type=int, default=1234, help="bootstrap seed")
    cmp_.add_argument(
        "--ignore-significance",
        action="store_true",
        help="fail on any regression past tolerance, even if it looks like noise",
    )
    cmp_.add_argument("--json", dest="json_out", help="write the comparison as JSON to this path")
    _add_normalizer_flags(cmp_)

    return parser


def _add_normalizer_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("normalization")
    group.add_argument("--keep-case", action="store_true", help="do not lowercase")
    group.add_argument("--keep-punctuation", action="store_true", help="do not strip punctuation")
    group.add_argument("--keep-contractions", action="store_true", help="do not expand contractions")
    group.add_argument("--drop-fillers", action="store_true", help="remove filler words before scoring")


def _normalizer_from_args(args: argparse.Namespace) -> NormalizerConfig:
    return NormalizerConfig(
        lowercase=not args.keep_case,
        strip_punctuation=not args.keep_punctuation,
        expand_contractions=not args.keep_contractions,
        drop_fillers=args.drop_fillers,
    )


def _load_corpus_score(path: str, config: NormalizerConfig) -> CorpusScore:
    """Load either a manifest (score it) or a previously written JSON result."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    if p.suffix.lower() == ".json":
        return _corpus_from_json(json.loads(p.read_text(encoding="utf-8")))
    return score_corpus(load_manifest(p), config)


def _corpus_from_json(data: dict[str, Any]) -> CorpusScore:
    """Rebuild a CorpusScore from the JSON `run --json` writes.

    Only the counts are restored; reference and hypothesis text are not needed
    for a comparison and are intentionally not required, so a baseline file can
    be committed to a repository without shipping the transcripts themselves.
    """
    scores: list[UtteranceScore] = []
    for entry in data.get("utterances", []):
        word = entry.get("word", {})
        char = entry.get("char", {})
        scores.append(
            UtteranceScore(
                utterance_id=str(entry["id"]),
                word=_counts_from_dict(word),
                char=_counts_from_dict(char),
                reference=entry.get("reference", ""),
                hypothesis=entry.get("hypothesis", ""),
            )
        )
    word_total = EditCounts()
    char_total = EditCounts()
    for s in scores:
        word_total = word_total + s.word
        char_total = char_total + s.char
    return CorpusScore(
        utterances=tuple(scores),
        word=word_total,
        char=char_total,
        normalizer=NormalizerConfig.from_dict(data.get("normalizer")),
    )


def _counts_from_dict(data: dict[str, Any]) -> EditCounts:
    return EditCounts(
        equal=int(data.get("equal", 0)),
        substitutions=int(data.get("substitutions", 0)),
        deletions=int(data.get("deletions", 0)),
        insertions=int(data.get("insertions", 0)),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = _normalizer_from_args(args)
    score = score_corpus(load_manifest(args.manifest), config)

    interval = None
    if not args.no_ci and score.count:
        interval = bootstrap_ci(
            score.utterances, resamples=args.resamples, seed=args.seed
        )

    if not args.quiet:
        print(render_score(score, interval=interval, worst=args.worst, show_alignment=args.alignment))

    if args.json_out:
        payload = score.to_dict()
        # Keep the text alongside the counts so a report is self-contained for a
        # human reader; `compare` never requires it.
        by_id = {u.utterance_id: u for u in score.utterances}
        for entry in payload["utterances"]:
            utt = by_id[entry["id"]]
            entry["reference"] = utt.reference
            entry["hypothesis"] = utt.hypothesis
        if interval is not None:
            payload["corpus"]["wer_interval"] = interval.to_dict()
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if not args.quiet:
            print(f"\nwrote {args.json_out}")

    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    config = _normalizer_from_args(args)
    baseline = _load_corpus_score(args.baseline, config)
    candidate = _load_corpus_score(args.candidate, config)

    gate = GateConfig(
        max_absolute_regression=args.max_absolute,
        max_relative_regression=args.max_relative,
        significance=args.significance,
        resamples=args.resamples,
        seed=args.seed,
        require_significance=not args.ignore_significance,
    )
    result = compare(baseline, candidate, gate)
    print(render_comparison(result))

    if args.json_out:
        payload = result.to_dict()
        payload["gate"] = gate.to_dict()
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")

    return EXIT_OK if result.passed else EXIT_GATE_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "compare":
            return _cmd_compare(args)
    except (FileNotFoundError, ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    parser.error(f"unknown command: {args.command}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
