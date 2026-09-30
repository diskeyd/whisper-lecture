"""Every run keeps its .review.jsonl sidecar in one flat repo directory, never in the vault."""

import sys
import types
from pathlib import Path

sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transcribe_whisper as tw


def test_vault_and_repo_runs_share_one_flat_dir():
    vault_md = tw.FILES_ROOT / "201.75 알고리즘" / "260930 알고리즘w5-1.md"
    assert tw.sidecar_path(vault_md) == tw.REVIEW_ROOT / "260930 알고리즘w5-1.review.jsonl"
    hyp_md = tw.REVIEW_ROOT.parent / "hyp" / "d3-ipad.md"
    assert tw.sidecar_path(hyp_md) == tw.REVIEW_ROOT / "d3-ipad.review.jsonl"


def test_write_creates_dir_and_overwrites_like_the_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(tw, "REVIEW_ROOT", tmp_path / "review")
    md = tw.FILES_ROOT / "201.62 시스템보안" / "x.md"
    tw.write_review_jsonl(md, [{"start": 0}, {"start": 1}])
    out = tw.write_review_jsonl(md, [{"start": 0}])
    assert out == tmp_path / "review" / "x.review.jsonl"
    assert len(out.read_text().splitlines()) == 1
