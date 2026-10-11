"""--plan says per recording where its transcript lands and whether the uploads are tried."""

import json
import os
import subprocess
import sys
import types
import unicodedata
from pathlib import Path

import pytest

sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import transcribe_whisper as tw

UUID = "0123abcd-0123-4567-89ab-0123456789ab"


@pytest.fixture
def vault(tmp_path, monkeypatch):
    subjects = tmp_path / "subjects"
    subjects.mkdir()
    (subjects / "201.62 시스템보안.md").write_text(
        f"- **NotebookLM**:: [NotebookLM](https://notebooklm.google.com/notebook/{UUID})\n",
        encoding="utf-8",
    )
    (subjects / "201.19 월간 Obsidian.md").write_text("no notebook here\n", encoding="utf-8")
    monkeypatch.setattr(tw, "SUBJECT_DIR", subjects)
    monkeypatch.setattr(tw, "FILES_ROOT", tmp_path / "raw")
    return tmp_path


def test_the_three_outcomes(vault):
    inbox = vault / "inbox"
    rows = tw.plan_rows(
        [
            str(inbox / "261008 시스템보안w6-2.m4a"),
            str(inbox / "월간obsidian 10월.m4a"),
            str(inbox / "Some YouTube talk.m4a"),
        ]
    )
    assert [(r["course"], r["dest"], r["notebook"]) for r in rows] == [
        ("201.62 시스템보안", str(vault / "raw" / "201.62 시스템보안"), True),
        ("201.19 월간 Obsidian", str(vault / "raw" / "201.19 월간 Obsidian"), False),
        (None, str(inbox), False),
    ]
    assert not (vault / "raw").exists()  # planning creates no course folder


def test_nfd_names_match_and_are_echoed_unchanged(vault):
    arg = unicodedata.normalize("NFD", str(vault / "261008 시스템보안w6-2.m4a"))
    (row,) = tw.plan_rows([arg])
    assert row["files"] == [arg]
    assert row["course"] == "201.62 시스템보안" and row["name"] == "261008 시스템보안w6-2.m4a"


def test_split_parts_plan_as_their_merged_file(vault):
    rows = tw.plan_rows([str(vault / "261001 시스템보안w5(2).m4a"), str(vault / "261001 시스템보안w5(1).m4a")])
    assert [(r["name"], len(r["files"])) for r in rows] == [("261001 시스템보안w5.m4a", 2)]
    assert rows[0]["files"][0].endswith("w5(1).m4a")


def test_unreadable_subject_notes_mean_unknown_not_no(vault, monkeypatch):
    monkeypatch.setattr(tw, "SUBJECT_DIR", vault / "missing")
    rows = tw.plan_rows([str(vault / "261008 시스템보안w6-2.m4a"), str(vault / "talk.m4a")])
    assert [r["notebook"] for r in rows] == [None, False]


def test_plan_mode_never_imports_mlx(tmp_path):
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "mlx_whisper.py").write_text("raise RuntimeError('mlx imported')\n")
    out = subprocess.run(
        [sys.executable, str(ROOT / "transcribe_whisper.py"), "--plan", str(tmp_path / "talk.m4a")],
        env={**os.environ, "PYTHONPATH": str(poison)},
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    (row,) = [json.loads(line) for line in out.splitlines()]
    assert row["course"] is None and row["notebook"] is False and out.isascii()
