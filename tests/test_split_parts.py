"""Split recordings passed together merge into one stem; a lone part warns about its siblings."""

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transcribe_whisper as tw

LISTED = ["261001 운영체제w5(2)", "261001 운영체제w5(1)", "261001 운영체제w5", "260930 알고리즘w5-1"]


def test_parts_passed_together_merge_in_number_order():
    groups, missing = tw.group_split_parts(LISTED[:2], LISTED)
    assert groups == [("261001 운영체제w5", ["261001 운영체제w5(1)", "261001 운영체제w5(2)"])]
    assert missing == []


def test_lone_part_stays_single_and_names_its_sibling():
    groups, missing = tw.group_split_parts(["261001 운영체제w5(2)"], LISTED)
    assert groups == [("261001 운영체제w5(2)", ["261001 운영체제w5(2)"])]
    assert missing == ["261001 운영체제w5(1)"]


def test_unnumbered_stems_never_merge():
    groups, missing = tw.group_split_parts(["261001 운영체제w5", "260930 알고리즘w5-1"], LISTED)
    assert [stems for _, stems in groups] == [["261001 운영체제w5"], ["260930 알고리즘w5-1"]]
    assert missing == []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_merge_concatenates_parts_into_one_file(tmp_path):
    parts = []
    for n, secs in ((2, 2), (1, 1)):  # passed out of order on purpose
        part = tmp_path / f"o'w5({n}).m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=d={secs}", "-c:a", "aac", str(part)],
            check=True,
        )
        parts.append(part)
    merged = tw.merge_split_recordings(parts)
    dst = tmp_path / "o'w5.m4a"
    assert merged == {dst: [parts[1], parts[0]]}
    assert 2.9 < tw.get_duration(dst) < 3.2
    assert not list(tmp_path.glob(".*merging*"))
