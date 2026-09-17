"""Tests for the openjev CLI argument handling and version output."""

import pytest

from openjev import __version__
from openjev.cli import main


def test_version(capsys):
    """`openjev --version` exits 0 and prints `openjev <version>`."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"openjev {__version__}" in capsys.readouterr().out


def test_decide_requires_preset_or_schema_and_context(capsys):
    """`decide` with neither --preset nor --schema/--context exits 2, naming what is missing."""
    with pytest.raises(SystemExit) as exc:
        main(["decide"])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "--preset" in stderr
    assert "--schema" in stderr
    assert "--context" in stderr


def test_decide_schema_without_context(capsys):
    """`decide --schema FILE` without --context exits 2, naming --context."""
    with pytest.raises(SystemExit) as exc:
        main(["decide", "--schema", "schema.json"])
    assert exc.value.code == 2
    assert "--context" in capsys.readouterr().err


def test_decide_context_without_schema(capsys):
    """`decide --context FILE` without --schema exits 2, naming --schema."""
    with pytest.raises(SystemExit) as exc:
        main(["decide", "--context", "context.txt"])
    assert exc.value.code == 2
    assert "--schema" in capsys.readouterr().err


def test_decide_rejects_preset_with_schema(capsys):
    """`decide --preset x --schema y` exits 2."""
    with pytest.raises(SystemExit) as exc:
        main(["decide", "--preset", "fintech_fraud", "--schema", "schema.json"])
    assert exc.value.code == 2


def test_decide_rejects_preset_with_context(capsys):
    """`decide --preset x --context y` exits 2."""
    with pytest.raises(SystemExit) as exc:
        main(["decide", "--preset", "fintech_fraud", "--context", "context.txt"])
    assert exc.value.code == 2
