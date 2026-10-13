"""Self-checks for judge_sheet. Synthetic text only — no lecture content."""

import judge_sheet

WHISPER = "**0:00 - 1:00**\n칠 더하기 일은 영 입니다 그리고 스택 하나리를 봅니다\n"
QWEN = "**0:00**\n7 더하기 일은 영입니다 그리고\n\n**0:20**\n스택 카나리를 봅니다\n"


def setup(tmp_path, monkeypatch):
    hyp, judge = tmp_path / "hyp", tmp_path / "judge"
    hyp.mkdir()
    (hyp / "x.current.md").write_text(WHISPER, encoding="utf-8")
    (hyp / "x.qwen.md").write_text(QWEN, encoding="utf-8")
    monkeypatch.setattr(judge_sheet, "HYP", hyp)
    monkeypatch.setattr(judge_sheet, "JUDGE", judge)
    return judge


def test_only_real_disagreements_are_listed(tmp_path, monkeypatch):
    judge = setup(tmp_path, monkeypatch)
    judge_sheet.make("x")
    sheet = (judge / "x.md").read_text(encoding="utf-8")
    # 칠/7 and 영 입니다/영입니다 differ only in number form and spacing
    assert sheet.count("- 판정:") == 1
    assert "- W: 하나리를" in sheet and "- Q: 카나리를" in sheet
    # second word of the last chunk (0:20, assumed 30 s long, 3 words) → 0:20 + 30 * 1/3
    assert "## 1 · 0:30" in sheet


def test_score_counts_wins_and_builds_reference(tmp_path, monkeypatch):
    judge = setup(tmp_path, monkeypatch)
    judge_sheet.make("x")
    path = judge / "x.md"
    path.write_text(path.read_text(encoding="utf-8").replace("- 판정: ", "- 판정: Q"), encoding="utf-8")
    tally, cer_w, cer_q = judge_sheet.score_one("x")
    assert tally["Q"] == 1 and tally["W"] == 0
    assert cer_q == 0 and cer_w > 0


def test_free_text_is_read_as_heard_text(tmp_path, monkeypatch):
    judge = setup(tmp_path, monkeypatch)
    judge_sheet.make("x")
    path = judge / "x.md"
    sheet = path.read_text(encoding="utf-8")
    # context words around the span are trimmed, then the text matches Qwen
    path.write_text(sheet.replace("- 판정: ", "- 판정: 그리고 스택 카나리를"), encoding="utf-8")
    assert judge_sheet.score_one("x")[0]["Q"] == 1
    path.write_text(sheet.replace("- 판정: ", "- 판정: 둘다 똑같은데"), encoding="utf-8")
    assert judge_sheet.score_one("x")[0]["B"] == 1
    path.write_text(sheet.replace("- 판정: ", "- 판정: 하나리"), encoding="utf-8")
    tally = judge_sheet.score_one("x")[0]
    assert tally["X"] == 1 and tally["XW"] == 1
    path.write_text(sheet.replace("- 판정: ", "- 판정: ? 문서로 확인"), encoding="utf-8")
    tally, cer_w, cer_q = judge_sheet.score_one("x")
    assert tally[""] == 1 and cer_w == cer_q == 0
