"""Guards on the clip coverage check.

The clips were cut at vault block stamps whose clock drifts by minutes, so the
question "is the word actually in this clip" is decided by content alignment,
not by the filename. Two things have to hold for that answer to be worth
anything: the index has to be read whole (a clip can carry two candidates, and
the filename shows only one), and a word touching the cut boundary must not be
reported as covered — the owner would hear half of it and judge on that.
"""

import clip_check as cc

TOC = """# 검증 클립 목차

## 01. `01_BBFT_38m00s.m4a`

- 들을 것: **BBFT → BFT**
- 원본: 260514 보안기술최신동향w11-2.md  (38:00 구간)

## 04. `04_프로로그_02m00s.m4a`

- 들을 것: **프로로그 → 프롤로그**
- 들을 것: **함수의 프로로그 → 함수의 프롤로그**
- 원본: 260908 시스템보안w2-1(1).md  (2:00 구간)

## 23. `23_ready키_156m00s.m4a`

- 들을 것: **ready키 → ready 큐**
- 원본: 260915 시스템보안FL04-2.md  (02:36:00 구간)
"""


def test_toc_keeps_both_candidates_of_one_clip(tmp_path):
    path = tmp_path / "00_목차.md"
    path.write_text(TOC, encoding="utf-8")
    entries, skipped = cc.parse_toc(path)
    assert skipped == []

    assert [e["num"] for e in entries] == ["01", "04", "23"]
    assert entries[1]["cands"] == [
        ("프로로그", "프롤로그"),
        ("함수의 프로로그", "함수의 프롤로그"),
    ]
    assert entries[0]["source"] == "260514 보안기술최신동향w11-2.md"


def test_stamp_reads_both_clock_formats():
    # The shipped transcripts mix "**59:00 - 60:00**" with "**01:00:00 - 01:01:00**".
    assert cc.stamp_seconds("38:00") == 2280
    assert cc.stamp_seconds("02:36:00") == 9360
    assert cc.stamp_seconds("127:00") == 7620


def test_judge_separates_covered_from_edge_and_missing():
    pad = "가" * 40
    hay = pad + "위협모델" + pad
    lo, hi = 0, len(hay)

    assert cc.judge(hay, lo, hi, "위협 모델")[0] == "ok"
    assert cc.judge(hay, lo, hi, "스택카나리")[0] == "miss"
    # Same word, but the cut stops right after it: half a phrase to listen to.
    assert cc.judge(hay, lo, len(pad) + 4 + 2, "위협 모델")[0] == "edge"
    assert cc.judge(hay, len(pad) - 2, hi, "위협 모델")[0] == "edge"


def test_judge_only_looks_inside_the_covered_stretch():
    hay = "위협모델" + "나" * 60 + "스택카나리"
    # The clip covers the tail only; the word earlier in the file is not in it.
    assert cc.judge(hay, 60, len(hay), "위협 모델")[0] == "miss"


def test_rename_keeps_number_and_word_and_takes_the_measured_time():
    assert cc.renamed("01_BBFT_38m00s.m4a", 2477) == "01_BBFT_41m17s.m4a"
    assert cc.renamed("05_함수의_프로로그_09m00s.m4a", 545) == "05_함수의_프로로그_09m05s.m4a"
    assert cc.renamed("22_인터롭트_127m00s.m4a", 7650) == "22_인터롭트_127m30s.m4a"


def test_repeated_word_in_one_window_picks_the_expected_occurrence():
    # A word worth checking is one the lecturer repeats: 펀테이너 sits in both the
    # 3:00 and the 4:00 block, and taking the first match cut both clips at 3:52.
    segments = [
        {"start": 10.0, "text": "펀테이너를 먼저 설명하고"},
        {"start": 40.0, "text": "다음 주제로 넘어가서"},
        {"start": 70.0, "text": "이 펀테이너 엔진은 따로 돌고"},
    ]
    assert cc.segment_time(segments, "펀테이너", "", near=12) == 10.0
    assert cc.segment_time(segments, "펀테이너", "", near=68) == 70.0
    # No expectation given: first match, the old behaviour.
    assert cc.segment_time(segments, "펀테이너", "") == 10.0
    assert cc.segment_time(segments, "없는말", "", near=12) is None


def test_context_outranks_timing_when_occurrences_are_a_minute_apart():
    # The drift comes from block labels a minute wide, so for 펀테이너 in the 3:00
    # and the 4:00 block it points at the wrong one as often as the right one.
    # The vault text around the occurrence the clip was cut for is what separates them.
    segments = [
        {"start": 10.0, "text": "컨테이너 사용법을 짧게 보고 펀테이너 얘기를 먼저 하겠습니다"},
        {"start": 70.0, "text": "이 펀테이너 엔진이 따로 돌고 하이퍼바이저와 구분됩니다"},
    ]
    # span scaled to this sample; in a real 4-minute window CONTEXT_SPAN is small
    # against ~1,200 characters of speech and the two slices barely overlap.
    engine = cc.normalize("펀테이너 엔진이 따로 돌고 하이퍼바이저")
    assert cc.segment_time(segments, "펀테이너", engine, near=12, span=14) == 70.0
    first = cc.normalize("컨테이너 사용법을 짧게 보고 펀테이너 얘기를 먼저")
    assert cc.segment_time(segments, "펀테이너", first, near=68, span=14) == 10.0


def test_context_wins_even_when_only_one_spelling_survives_the_rerun():
    # The re-run heard the word right at the occurrence this clip is for and wrong
    # at the earlier one, so the only direct match in the window is the wrong one.
    segments = [
        {"start": 10.0, "text": "여기서 스펙 스매싱이라는 기법을 짧게 설명하고 넘어가겠습니다"},
        {"start": 70.0, "text": "제어 흐름을 바꾸는 기법을 순서대로 적으면스택 스매싱 다음에 리턴 투 립씨로 이어집니다"},
    ]
    context = cc.normalize(
        "제어 흐름을 바꾸는 기법을 순서대로 적으면스펙 스매싱 다음에 리턴 투 립씨로 이어집니다"
    )
    assert cc.segment_time(segments, "스펙 스매싱", context, near=12, span=20) == 70.0


def test_covered_widens_past_the_run_up_the_two_decodes_disagree_on():
    # The shipped transcript wrote 컨테이너 where the re-run wrote 펀테이너, so nothing
    # in the run-up matches and only the tail aligns. Judging on matched text alone
    # puts the candidate outside the clip and calls a good clip broken.
    tail = cc.normalize("실무에서 자주 쓰는 방식이라고만 알아 두시면 됩니다")
    hay = cc.normalize("실습 환경도 요즘 펀테이너로 꾸리잖아요 그래서") + tail
    needle = cc.normalize("대회 환경도 개발 환경도 펀테이너로 꾸리는데") + tail

    # Widening is an estimate — the two decodes differ in length as well as in
    # wording — so the claim is that the word falls inside, not where exactly.
    lo, hi, _ = cc.covered(hay, needle)
    assert cc.judge(hay, lo, hi, "펀테이너")[0] != "miss"
    # Without it the match starts at the tail, past the word, and the clip is
    # reported broken although the word is audible in it.
    matched_lo, matched_hi = cc.locate(hay, needle)[:2]
    assert cc.judge(hay, matched_lo, matched_hi, "펀테이너")[0] == "miss"


def test_toc_skips_a_malformed_entry_instead_of_taking_the_run_down(tmp_path):
    """A whole pass costs an hour of decoding; one hand-edited line must not spend it."""
    path = tmp_path / "00_목차.md"
    path.write_text(
        TOC + "\n## 99. `99_깨진항목_00m00s.m4a`\n\n- 들을 것: **가나 → 다라**\n"
        "- 원본: 260909 시스템보안FL02.md  (없는시각 구간)\n",
        encoding="utf-8",
    )
    entries, skipped = cc.parse_toc(path)
    assert [e["num"] for e in entries] == ["01", "04", "23"]
    assert skipped == [("99", "99_깨진항목_00m00s.m4a")]


def test_mixed_clip_is_reported_not_recut():
    """Clips 04 and 31 carry two candidates. Recutting is per clip and deletes the
    old file, so recutting for one candidate would drop a clip that covers the
    other one correctly."""
    rows = [
        {"clip": "04_a.m4a", "cand": "프로로그", "status": "ok"},
        {"clip": "04_a.m4a", "cand": "함수의 프로로그", "status": "miss"},
    ]
    recut, manual = cc.plan_recuts(rows)
    assert recut == []
    assert [c for c, _ in manual] == ["04_a.m4a"]


def test_clip_with_every_candidate_bad_is_recut_once():
    rows = [
        {"clip": "04_a.m4a", "cand": "프로로그", "status": "miss"},
        {"clip": "04_a.m4a", "cand": "함수의 프로로그", "status": "edge"},
        {"clip": "05_b.m4a", "cand": "인터럴트", "status": "ok"},
    ]
    recut, manual = cc.plan_recuts(rows)
    assert [c for c, _ in recut] == ["04_a.m4a"]  # one entry, so one cut and one rename
    assert manual == []


def test_index_filenames_are_swapped_even_when_the_file_holds_them_decomposed(tmp_path):
    """iCloud hands filenames back as NFD, so the index can hold either form."""
    import unicodedata

    old, new = "09_인터럴트_07m00s.m4a", "09_인터럴트_07m16s.m4a"
    for form in ("NFC", "NFD"):
        d = tmp_path / form
        d.mkdir()
        (d / cc.TOC).write_text(
            f"## 09. `{unicodedata.normalize(form, old)}`\n", encoding="utf-8"
        )
        (d / cc.ORDER).write_text(
            f"- 클립: `{unicodedata.normalize(form, old)}`\n", encoding="utf-8"
        )
        cc.patch_docs(d, {old: new})
        for name in (cc.TOC, cc.ORDER):
            assert new in (d / name).read_text(encoding="utf-8"), form
