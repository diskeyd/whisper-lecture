"""--no-merge: every path is its own recording, even when numbered siblings sit beside it."""

import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transcribe_whisper as tw


@pytest.fixture
def plan_env(tmp_path, monkeypatch):
    monkeypatch.setattr(tw, "SUBJECT_DIR", tmp_path / "subjects")
    monkeypatch.setattr(tw, "FILES_ROOT", tmp_path / "raw")
    return tmp_path


def test_plan_gives_one_row_per_part(plan_env):
    parts = [str(plan_env / "261001 시스템보안w5(2).m4a"), str(plan_env / "261001 시스템보안w5(1).m4a")]
    rows = tw.plan_rows(parts, no_merge=True)
    assert [(r["name"], r["files"]) for r in rows] == [
        ("261001 시스템보안w5(2).m4a", [parts[0]]),
        ("261001 시스템보안w5(1).m4a", [parts[1]]),
    ]


def test_plan_without_the_flag_still_groups(plan_env):
    rows = tw.plan_rows([str(plan_env / "w5(1).m4a"), str(plan_env / "w5(2).m4a")])
    assert [(r["name"], len(r["files"])) for r in rows] == [("w5.m4a", 2)]


@pytest.mark.parametrize("extra", [[], ["--keep-audio"], ["--no-upload"]])
def test_main_skips_the_merge_step(tmp_path, monkeypatch, extra):
    part, sibling = tmp_path / "X(1).m4a", tmp_path / "X(2).m4a"
    part.write_bytes(b"")
    sibling.write_bytes(b"")
    seen = []
    monkeypatch.setattr(tw, "SUBJECT_DIR", tmp_path / "subjects")  # uploading runs need the vault set

    def no_merge_allowed(paths):
        raise AssertionError("merge_split_recordings called")

    monkeypatch.setattr(tw, "merge_split_recordings", no_merge_allowed)
    monkeypatch.setattr(tw, "check_dependencies", lambda: None)
    monkeypatch.setattr(tw, "choose_language", lambda *a: None)
    monkeypatch.setattr(
        tw, "transcribe_file", lambda path, **kw: (seen.append((path.name, kw["keep_audio"], kw["upload"])), (True, "ok"))[1]
    )
    monkeypatch.setattr(sys, "argv", ["transcribe_whisper.py", "--no-merge", *extra, str(part)])
    tw.main()
    assert seen == [("X(1).m4a", "--keep-audio" in extra, "--no-upload" not in extra)]
    assert sibling.exists()


def test_main_still_merges_without_the_flag(tmp_path, monkeypatch):
    part = tmp_path / "X(1).m4a"
    part.write_bytes(b"")
    called = []
    monkeypatch.setattr(tw, "merge_split_recordings", lambda paths: called.append(paths) or {p: [p] for p in paths})
    monkeypatch.setattr(tw, "check_dependencies", lambda: None)
    monkeypatch.setattr(tw, "choose_language", lambda *a: None)
    monkeypatch.setattr(tw, "transcribe_file", lambda path, **kw: (True, "ok"))
    monkeypatch.setattr(sys, "argv", ["transcribe_whisper.py", "--no-upload", str(part)])
    tw.main()
    assert called
