"""Text normalization applied before scoring.

Normalization is the single most under-documented source of disagreement
between two WER numbers. Two systems evaluated with different normalizers are
not comparable, so the configuration is explicit, serializable, and recorded
in every report this package emits.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

# Fillers are removed only when `drop_fillers` is enabled. Whether a filler is
# an error is a product decision, not a universal truth: a captioning product
# usually wants them gone, a conversational-analytics product may want them.
DEFAULT_FILLERS: tuple[str, ...] = (
    "uh",
    "uhm",
    "um",
    "erm",
    "hmm",
    "mhm",
    "eh",
    "ah",
)

# Expanded rather than guessed at scoring time, so "don't" and "do not" do not
# register as a substitution purely because of transcription house style.
DEFAULT_CONTRACTIONS: dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}

_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


@dataclass(frozen=True)
class NormalizerConfig:
    """Declarative description of a normalization pipeline."""

    lowercase: bool = True
    strip_punctuation: bool = True
    expand_contractions: bool = True
    drop_fillers: bool = False
    collapse_whitespace: bool = True
    unicode_form: str = "NFKC"
    fillers: tuple[str, ...] = field(default=DEFAULT_FILLERS)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fillers"] = list(self.fillers)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NormalizerConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "fillers" in kwargs and kwargs["fillers"] is not None:
            kwargs["fillers"] = tuple(kwargs["fillers"])
        return cls(**kwargs)


class Normalizer:
    """Applies a `NormalizerConfig` to raw transcript text."""

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()
        self._fillers = frozenset(f.lower() for f in self.config.fillers)

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """Return `text` with the configured transformations applied."""
        if text is None:
            return ""
        out = unicodedata.normalize(self.config.unicode_form, str(text))
        if self.config.lowercase:
            out = out.lower()
        if self.config.expand_contractions:
            out = self._expand_contractions(out)
        if self.config.strip_punctuation:
            out = _PUNCT_RE.sub(" ", out)
            out = out.replace("'", "")
        if self.config.collapse_whitespace:
            out = _WS_RE.sub(" ", out).strip()
        if self.config.drop_fillers:
            out = " ".join(t for t in out.split() if t not in self._fillers)
        return out

    def tokenize(self, text: str) -> list[str]:
        """Normalize `text` and split it into word tokens."""
        normalized = self.normalize(text)
        return normalized.split() if normalized else []

    def characters(self, text: str) -> list[str]:
        """Normalize `text` and split it into characters, spaces excluded."""
        normalized = self.normalize(text)
        return [c for c in normalized if not c.isspace()]

    @staticmethod
    def _expand_contractions(text: str) -> str:
        out = text
        for src, dst in DEFAULT_CONTRACTIONS.items():
            out = out.replace(src, dst)
        return out


def normalize_all(texts: Iterable[str], config: NormalizerConfig | None = None) -> list[str]:
    """Convenience helper: normalize an iterable of strings with one config."""
    normalizer = Normalizer(config)
    return [normalizer.normalize(t) for t in texts]
