# asr-eval

Evaluation and regression-gating toolkit for speech recognition systems.

Scores ASR output against reference transcripts, reports where the errors are,
and decides whether a change is safe to ship — with a bootstrap test so the gate
fires on real regressions instead of on sampling noise.

Pure Python standard library. No runtime dependencies, no model downloads, no
API keys.

```
$ asr-eval run examples/sample_manifest.jsonl
==============================================================
ASR EVALUATION
==============================================================
utterances       12
reference words  93
WER              0.0430  [0.0110, 0.0761] at 95%
CER              0.0342

substitutions 2   deletions 1   insertions 1
error mix     S 50%   D 25%   I 25%

--- worst 3 utterances by error count ---
[utt-0010] WER 0.1429  (1 err / 7 ref words)
  REF: send the report to the engineering team
  HYP: send the report to the engineering team today
```

## Why this exists

Most ASR evaluation in practice is a WER number in a notebook. That number is
enough to tell you a model changed and not much else, and it invites the most
common mistake in model evaluation: treating a 0.3-point move on a 200-utterance
set as a regression, then rolling back a model that was fine.

This tool takes three positions:

**Normalization is part of the metric.** Two systems scored with different
normalizers are not comparable. The normalizer configuration is explicit,
serializable, recorded in every report, and checked during comparison — if the
two runs disagree, the report says so rather than quietly reporting a delta.

**A number without an interval is not a result.** Corpus WER is reported with a
bootstrap confidence interval, resampled over utterances rather than tokens,
because errors within an utterance are correlated and token-level resampling
understates the interval.

**A gate that fires on noise gets bypassed.** The release gate blocks only when
a regression clears an absolute tolerance _and_ a paired bootstrap says it is
unlikely to be sampling noise. Either check alone produces gates teams learn to
ignore within a month, and a bypassed gate protects nothing.

## Install

```bash
git clone https://github.com/<your-username>/asr-eval.git
cd asr-eval
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Usage

### Score a run

Input is a JSON Lines manifest — one utterance per line, so a corpus can be
appended to without rewriting the file:

```jsonl
{"id": "utt-0001", "reference": "the quick brown fox", "hypothesis": "the quick brown fox"}
{"id": "utt-0002", "reference": "can you hear me now", "hypothesis": "can you hear me"}
```

Any extra keys per line (speaker, duration, locale, audio path) are preserved as
metadata.

```bash
asr-eval run manifest.jsonl                      # human-readable report
asr-eval run manifest.jsonl --json result.json   # machine-readable result
asr-eval run manifest.jsonl --alignment          # show token-level alignments
asr-eval run manifest.jsonl --drop-fillers       # ignore um / uh / erm
```

### Gate a release

Score a known-good run once and commit it as your baseline:

```bash
asr-eval run baseline_manifest.jsonl --json baseline.json
git add baseline.json
```

Then compare every candidate against it. The command exits `1` when the gate
fails, so CI blocks without any extra scripting:

```bash
asr-eval compare baseline.json candidate_manifest.jsonl
```

```
==============================================================
ASR REGRESSION GATE: FAIL
==============================================================
compared utterances  12
baseline WER         0.0430
candidate WER        0.1720
delta                +0.1290 (+300.00% regression)
paired bootstrap p   0.0010

notes:
  - WER regressed by 0.1290 absolute, above the 0.0050 tolerance
  - paired bootstrap p=0.0010 < 0.05
```

Tuning the gate:

```bash
--max-absolute 0.005        # absolute WER tolerance (default: 0.5 points)
--max-relative 0.02         # optional relative tolerance
--significance 0.05         # paired-bootstrap alpha
--ignore-significance       # block on any regression past tolerance
```

A baseline JSON stores only counts, not transcripts, so it can be committed to a
repository without shipping the reference data itself.

### As a library

```python
from asr_eval import load_manifest, score_corpus, bootstrap_ci
from asr_eval.compare import GateConfig, compare

score = score_corpus(load_manifest("run.jsonl"))
print(score.wer, score.cer)
print(bootstrap_ci(score.utterances).to_dict())

for utt in score.worst(10):
    print(utt.utterance_id, utt.wer, utt.reference, utt.hypothesis)

result = compare(baseline_score, score, GateConfig(max_absolute_regression=0.01))
if not result.passed:
    raise SystemExit("\n".join(result.reasons))
```

## What the report tells you

The error mix is the part worth reading first. The three error classes usually
point at three different root causes:

| Signal                              | Common cause                                                     |
| ----------------------------------- | ---------------------------------------------------------------- |
| Deletion spike                      | Truncation, endpointing cutting audio short, VAD dropping speech |
| Insertion spike on clean references | Hallucinated output, decoder not terminating                     |
| Substitution spike                  | Vocabulary drift, accent or domain mismatch, number formatting   |

`worst` ranks by error _count_ before error rate, because ranking by rate alone
fills the list with three-word utterances that got one word wrong.

## CI integration

`.github/workflows/ci.yml` runs the test suite on Python 3.10–3.12 and then runs
the gate itself against the committed baseline, uploading the report as a build
artifact. The gate step is the tool doing the job it exists for.

```yaml
- name: Score the current run
  run: asr-eval run examples/sample_manifest.jsonl --json current.json

- name: Gate against the committed baseline
  run: asr-eval compare examples/baseline.json current.json
```

## Design notes

**Corpus WER aggregates before dividing.** Total errors over total reference
tokens, not the mean of per-utterance rates. The two differ whenever utterance
lengths differ, and the unweighted mean lets a handful of very short utterances
dominate the headline number.

**Empty references have an explicit rule.** No reference tokens means no
denominator. An empty hypothesis scores 0.0 (nothing expected, nothing produced)
and a non-empty one scores 1.0 (everything produced is an insertion), so the
corpus aggregate stays finite without silently dropping the utterance.

**Comparisons run on the shared subset.** If two runs cover different utterance
sets, scoring them against each other changes the denominator and produces a
"regression" that is really a change of test set. The overlap is compared and
the difference is reported.

**Alignment ties resolve deterministically** — substitution, then deletion, then
insertion. A gate that reports different counts on identical input is worse than
no gate.

**Bootstrap seeds are fixed by default.** A gate whose verdict changes between
two identical runs cannot be trusted to block a release.

## Tests

```bash
pytest -q     # 65 tests
```

Coverage includes the alignment edge cases (empty sequences on either side, tie
determinism, index mapping), the empty-reference rule, corpus aggregation vs.
per-utterance means, bootstrap reproducibility, gate behaviour on mismatched
utterance sets and mismatched normalizers, and the CLI exit codes.

## Roadmap

- Streaming metrics: time-to-first-token and partial-hypothesis stability
- Per-slice reporting (speaker, locale, duration bucket, SNR) from manifest metadata
- Text-normalization presets per domain (medical, finance, telephony)
- Confusion-pair extraction to surface systematic substitutions

## License

MIT — see [LICENSE](LICENSE).
