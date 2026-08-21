import json
from pathlib import Path

import pytest

from asr_eval.cli import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, main


def write_manifest(path: Path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def clean_rows(n=30):
    return [{"id": f"u{i}", "reference": "the cat sat down", "hypothesis": "the cat sat down"} for i in range(n)]


def broken_rows(n=30):
    return [{"id": f"u{i}", "reference": "the cat sat down", "hypothesis": "a dog stood up"} for i in range(n)]


def test_run_prints_report_and_exits_zero(tmp_path, capsys):
    m = write_manifest(tmp_path / "m.jsonl", clean_rows(3))
    assert main(["run", str(m)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "ASR EVALUATION" in out
    assert "WER" in out


def test_run_writes_json(tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", broken_rows(5))
    out = tmp_path / "result.json"
    assert main(["run", str(m), "--json", str(out), "--quiet"]) == EXIT_OK
    payload = json.loads(out.read_text())
    assert payload["corpus"]["utterances"] == 5
    assert payload["corpus"]["wer"] > 0
    assert "normalizer" in payload
    assert payload["utterances"][0]["reference"]


def test_run_reports_missing_file(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.jsonl")]) == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_run_rejects_malformed_json(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"reference": "a", "hypothesis": "a"}\nnot json\n', encoding="utf-8")
    assert main(["run", str(bad)]) == EXIT_USAGE
    assert "invalid JSON" in capsys.readouterr().err


def test_run_rejects_missing_fields(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"reference": "a"}\n', encoding="utf-8")
    assert main(["run", str(bad)]) == EXIT_USAGE
    assert "missing required field" in capsys.readouterr().err


def test_compare_passes_on_identical_runs(tmp_path, capsys):
    m = write_manifest(tmp_path / "m.jsonl", clean_rows())
    base = tmp_path / "base.json"
    main(["run", str(m), "--json", str(base), "--quiet"])
    assert main(["compare", str(base), str(m)]) == EXIT_OK
    assert "PASS" in capsys.readouterr().out


def test_compare_fails_and_exits_nonzero_on_regression(tmp_path, capsys):
    good = write_manifest(tmp_path / "good.jsonl", clean_rows())
    bad = write_manifest(tmp_path / "bad.jsonl", broken_rows())
    base = tmp_path / "base.json"
    main(["run", str(good), "--json", str(base), "--quiet"])
    assert main(["compare", str(base), str(bad)]) == EXIT_GATE_FAILED
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "regressed" in out


def test_compare_accepts_two_json_files(tmp_path):
    good = write_manifest(tmp_path / "good.jsonl", clean_rows())
    bad = write_manifest(tmp_path / "bad.jsonl", broken_rows())
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    main(["run", str(good), "--json", str(a), "--quiet"])
    main(["run", str(bad), "--json", str(b), "--quiet"])
    assert main(["compare", str(a), str(b)]) == EXIT_GATE_FAILED


def test_compare_writes_json(tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", clean_rows())
    base = tmp_path / "base.json"
    out = tmp_path / "cmp.json"
    main(["run", str(m), "--json", str(base), "--quiet"])
    main(["compare", str(base), str(m), "--json", str(out)])
    payload = json.loads(out.read_text())
    assert payload["passed"] is True
    assert "gate" in payload


def test_drop_fillers_flag_changes_the_score(tmp_path):
    rows = [{"id": "u1", "reference": "um yes please", "hypothesis": "yes please"}]
    m = write_manifest(tmp_path / "m.jsonl", rows)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    main(["run", str(m), "--json", str(a), "--quiet"])
    main(["run", str(m), "--json", str(b), "--quiet", "--drop-fillers"])
    assert json.loads(a.read_text())["corpus"]["wer"] > 0
    assert json.loads(b.read_text())["corpus"]["wer"] == 0


def test_alignment_flag_renders_rows(tmp_path, capsys):
    m = write_manifest(tmp_path / "m.jsonl", broken_rows(2))
    main(["run", str(m), "--alignment", "--worst", "1"])
    assert "REF:" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "asr-eval" in capsys.readouterr().out
