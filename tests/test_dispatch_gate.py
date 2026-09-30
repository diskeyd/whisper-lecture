"""The original recording is deleted only when Drive has it and the NotebookLM source is ready.

The NotebookLM CLI is mocked. "processing until the cap" is the CLI's own timeout outcome
of `source wait`; the polling itself is the CLI's contract and is not exercised here.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transcribe_whisper as tw

WAIT = {
    "ready": (True, {"source_id": "s1", "status": "ready"}),
    "timeout": (False, {"source_id": "s1", "status": "timeout"}),
    "failed": (False, {"source_id": "s1", "status": "error"}),
}


def run(monkeypatch, tmp_path, *, add_id: str | None = "s1", wait="ready", listed=None):
    audio = tmp_path / "260930 알고리즘w5-1.m4a"
    audio.write_bytes(b"x")
    calls = []

    def fake_cli(cmd, timeout):
        calls.append(cmd[1:3])
        if cmd[1:3] == ["source", "list"]:
            return True, json.dumps({"sources": listed or []}), ""
        if cmd[1:3] == ["source", "add"]:
            return True, json.dumps({"source": {"id": add_id} if add_id else {}}), ""
        if cmd[1:3] == ["source", "wait"]:
            ok, body = WAIT[wait]
            return ok, json.dumps(body), "" if ok else "exit 1"
        raise AssertionError(cmd)

    monkeypatch.setattr(tw, "run_cli", fake_cli)
    monkeypatch.setattr(tw, "upload_to_drive", lambda p: ("d1", "업로드"))
    monkeypatch.setattr(tw, "find_notebook_uuid", lambda c: ("uuid", "note"))
    monkeypatch.setattr(tw, "relogin", lambda *a: False)
    ok, _ = tw.dispatch_audio(audio, "201.75 알고리즘")
    return ok, audio.exists(), calls


def test_ready_deletes(monkeypatch, tmp_path):
    ok, exists, calls = run(monkeypatch, tmp_path)
    assert ok and not exists
    assert ["source", "wait"] in calls


@pytest.mark.parametrize("wait", ["timeout", "failed"])
def test_not_ready_keeps(monkeypatch, tmp_path, wait):
    ok, exists, _ = run(monkeypatch, tmp_path, wait=wait)
    assert not ok and exists


def test_missing_source_id_keeps(monkeypatch, tmp_path):
    ok, exists, calls = run(monkeypatch, tmp_path, add_id=None)
    assert not ok and exists
    assert ["source", "wait"] not in calls


def test_upload_only_retry_waits_on_processing_source(monkeypatch, tmp_path):
    listed = [{"id": "s1", "title": "260930 알고리즘w5-1.m4a", "status": "processing"}]
    ok, exists, calls = run(monkeypatch, tmp_path, listed=listed, wait="timeout")
    assert not ok and exists
    assert ["source", "add"] not in calls
    ok, exists, _ = run(monkeypatch, tmp_path, listed=listed, wait="ready")
    assert ok and not exists
