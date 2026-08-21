"""Manifest loading.

The manifest is JSON Lines: one utterance per line, so a corpus can be streamed
and appended to without rewriting the file, and a malformed line names its own
line number instead of invalidating the whole run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


class ManifestError(ValueError):
    """Raised when a manifest line is missing required fields or is malformed."""


@dataclass(frozen=True)
class Utterance:
    """One scored unit: a reference transcript and a system hypothesis."""

    utterance_id: str
    reference: str
    hypothesis: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "<memory>", line: int = 0) -> "Utterance":
        missing = [k for k in ("reference", "hypothesis") if k not in data]
        if missing:
            raise ManifestError(
                f"{source}:{line}: missing required field(s): {', '.join(missing)}"
            )
        known = {"id", "utterance_id", "reference", "hypothesis"}
        utt_id = data.get("utterance_id") or data.get("id") or f"utt-{line:06d}"
        metadata = {k: v for k, v in data.items() if k not in known}
        return cls(
            utterance_id=str(utt_id),
            reference=str(data["reference"]),
            hypothesis=str(data["hypothesis"]),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.utterance_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
        }
        data.update(self.metadata)
        return data


def iter_manifest(path: str | Path) -> Iterator[Utterance]:
    """Yield utterances from a JSON Lines manifest, skipping blank lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
            if not isinstance(payload, dict):
                raise ManifestError(f"{path}:{line_no}: expected a JSON object")
            yield Utterance.from_dict(payload, source=str(path), line=line_no)


def load_manifest(path: str | Path) -> list[Utterance]:
    """Load an entire JSON Lines manifest into memory."""
    return list(iter_manifest(path))
