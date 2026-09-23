"""Self-checks for the measurement tools. Synthetic text only — no lecture content."""

import json
import unicodedata

import pytest
from cer import cer, cer_canonical
from review_sheet import render
from slice_transcript import align
from term_error import count_terms, omission_rate, term_error_rate
from textnorm import (
    canonicalize,
    load_aliases,
    load_body,
    load_glossary,
    normalize,
    replace_guarded,
)

GOLD = """**0:00 - 1:00**

스택 카나리는 버퍼 오버플로를 탐지하는 값입니다.

**1:00 - 2:00**

플로우 컨트롤은 수신 버퍼의 여유를 알려 줍니다.
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ─── the ruler must read zero on itself ──────────────────────────────────────


def test_cer_gold_vs_gold_is_zero(tmp_path):
    gold = write(tmp_path, "gold.md", GOLD)
    assert cer(load_body(gold), load_body(gold)) == pytest.approx(0.0)


def test_cer_nfd_copy_of_gold_is_zero(tmp_path):
    """macOS/iCloud hands back decomposed Hangul; without NFC this inflates badly."""
    gold = write(tmp_path, "gold.md", GOLD)
    nfd = write(tmp_path, "gold-nfd.md", unicodedata.normalize("NFD", GOLD))
    assert nfd.read_text(encoding="utf-8") != gold.read_text(encoding="utf-8")
    assert cer(load_body(gold), load_body(nfd)) == pytest.approx(0.0)


def test_cer_catches_a_real_error(tmp_path):
    gold = write(tmp_path, "gold.md", GOLD)
    hyp = write(tmp_path, "hyp.md", GOLD.replace("스택 카나리", "스택 하나리"))
    assert cer(load_body(gold), load_body(hyp)) > 0.0


# ─── headers and punctuation must not reach the metrics ──────────────────────


def test_headers_are_dropped_and_do_not_count(tmp_path):
    """The shipped slice keeps absolute times, a re-run starts at 0:00."""
    shipped = write(tmp_path, "shipped.md", GOLD.replace("0:00 - 1:00", "20:00 - 21:00")
                    .replace("1:00 - 2:00", "21:00 - 22:00"))
    current = write(tmp_path, "current.md", GOLD)
    assert "20:00" not in load_body(shipped)
    assert cer(load_body(shipped), load_body(current)) == pytest.approx(0.0)


def test_hour_format_header_is_dropped(tmp_path):
    path = write(tmp_path, "long.md", "**01:00:00 - 01:01:00**\n자 다음으로 넘어갑니다.\n")
    assert load_body(path).strip() == "자 다음으로 넘어갑니다."


def test_sheet_style_single_timestamps_are_dropped(tmp_path):
    """A gold file corrected from the sheet carries '**3:20**' in front of every
    segment; left in, those digits alone inflate CER by roughly a fifth."""
    plain = write(tmp_path, "gold.md", "스택 카나리는 버퍼 오버플로를 탐지합니다. 플로우 컨트롤은 별개입니다.")
    marked = write(tmp_path, "from-sheet.md",
                   "**0:05** 스택 카나리는 버퍼 오버플로를 탐지합니다.\n\n"
                   "**0:12** 플로우 컨트롤은 별개입니다.\n")
    assert "0:05" not in load_body(marked)
    assert cer(load_body(plain), load_body(marked)) == pytest.approx(0.0)


def test_corrector_notes_are_dropped(tmp_path):
    """The corrector leaves questions in the gold file; they are not speech."""
    plain = write(tmp_path, "plain.md", "부모가 자식들보다 큽니다.")
    noted = write(tmp_path, "noted.md",
                  "부모가 자식들보다 <차이들이라고 들리지만 자식일 것 같다> 큽니다.")
    assert "차이들" not in load_body(noted)
    assert cer(load_body(plain), load_body(noted)) == pytest.approx(0.0)


def test_normalize_drops_spacing_and_punctuation():
    assert normalize("플로우 컨트롤, 맞죠?") == normalize("플로우컨트롤 맞죠")
    assert normalize("「해시」") == normalize("해시")  # quotes the script's regex misses


# ─── term error rate ─────────────────────────────────────────────────────────


GLOSSARY_TSV = """# correct<TAB>wrong1|wrong2<TAB>course id
스택 카나리\t스택 하나리\t201.62 시스템보안
플로우 컨트롤\t\t
해시 함수\t해쉬 함수\t
"""


def test_glossary_scope_and_format(tmp_path):
    path = write(tmp_path, "terms.tsv", GLOSSARY_TSV)
    assert len(load_glossary(path)) == 3
    scoped = load_glossary(path, "201.66 컴퓨터네트워크")
    assert [c for c, _ in scoped] == ["플로우 컨트롤", "해시 함수"]
    assert load_glossary(tmp_path / "missing.tsv") == []


def test_term_error_counts_occurrences_not_terms(tmp_path):
    entries = load_glossary(write(tmp_path, "terms.tsv", GLOSSARY_TSV))
    gold = "스택 카나리와 플로우 컨트롤, 그리고 다시 플로우 컨트롤."
    hyp = "스택 하나리와 플로우 컨트롤, 그리고 다시 플로우 컨트롤."
    rate, missed, total = term_error_rate(gold, hyp, entries)
    assert (missed, total) == (1, 3)  # 3 occurrences in gold, 1 broken
    assert rate == pytest.approx(1 / 3)


def test_term_error_ignores_terms_the_lecture_never_said(tmp_path):
    entries = load_glossary(write(tmp_path, "terms.tsv", GLOSSARY_TSV))
    rate, missed, total = term_error_rate("해시 함수를 씁니다.", "해시 함수를 씁니다.", entries)
    assert (rate, missed, total) == (0.0, 0, 1)


def test_term_error_is_spacing_insensitive(tmp_path):
    entries = load_glossary(write(tmp_path, "terms.tsv", GLOSSARY_TSV))
    _, missed, total = term_error_rate("해시 함수", "해시함수", entries)
    assert (missed, total) == (0, 1)


# ─── omission rate ───────────────────────────────────────────────────────────


def test_omission_rate_counts_dropped_eojeol():
    gold = "자 여기서 플로우 컨트롤을 한번 보겠습니다"  # 6 eojeol
    hyp = "자 여기서 플로우 보겠습니다"  # 컨트롤을, 한번 dropped
    rate, deletions, n = omission_rate(gold, hyp)
    assert (deletions, n) == (2, 6)
    assert rate == pytest.approx(2 / 6)


def test_omission_rate_is_zero_on_identical_text():
    rate, deletions, _ = omission_rate(GOLD, GOLD)
    assert (rate, deletions) == (0.0, 0)


# ─── shipped transcript alignment ───────────────────────────────────────────

# The old transcripts' clocks drift (audio 48:00 sat under "**46:00 - 47:00**" in
# 260512 w11-1), so the shipped text is located by content, not by timestamp.
SHIPPED_FULL = (
    "앞쪽 내용이 길게 이어지는 구간입니다 여기는 발췌 밖입니다 "
    "완전 이진트리를 만들어 놓으면 정렬이 쉬워집니다 그래서 빌드 힙부터 봅니다 "
    "뒤쪽 내용이 이어지는 구간이고 여기도 발췌 밖입니다"
)
RERUN = "완전 이진트리를 만들어 놓으면 정렬이 쉬워집니다 그래서 빌드 팁부터 봅니다"


def test_align_finds_the_excerpt_span_not_the_nominal_minutes():
    cut, ratio = align(SHIPPED_FULL, RERUN)
    assert "완전 이진트리" in cut
    assert "발췌 밖" not in cut
    assert ratio > 0.8  # same speech, one word transcribed differently


def test_align_rejects_unrelated_text():
    with pytest.raises(SystemExit):
        align(SHIPPED_FULL, "전혀 다른 강의의 내용이고 겹치는 구절이 하나도 없습니다")


def test_align_survives_nfd_input():
    """The vault is on iCloud, which hands back decomposed Hangul."""
    cut, ratio = align(unicodedata.normalize("NFD", SHIPPED_FULL), RERUN)
    assert "완전 이진트리" in cut
    assert "발췌 밖" not in cut
    assert ratio > 0.8


# ─── correction sheet ────────────────────────────────────────────────────────


def test_sheet_puts_flagged_segments_before_the_draft():
    records = [
        {"start": 0.0, "end": 3.0, "text": "정상 구간입니다.", "flags": []},
        {"start": 3.0, "end": 6.0, "text": "의심 구간입니다.", "flags": ["low_logprob", "disagree"],
         "alt": "다른 결과입니다.", "avg_logprob": -0.9, "no_speech_prob": 0.1},
    ]
    sheet = render(records, "s1-261")
    assert sheet.index("의심 구간입니다.") < sheet.index("## 전체 초안")
    assert sheet.index("## 플래그 구간") < sheet.index("정상 구간입니다.")
    assert "낮은 확신" in sheet and "2차 디코딩 불일치" in sheet
    assert "다른 결과입니다." in sheet


def test_sheet_marks_a_dropped_segment():
    records = [{"start": 0.0, "end": 2.0, "text": "", "raw_text": "네 네 네 네",
                "flags": ["repetitive"]}]
    sheet = render(records, "s2-262")
    assert "(삭제됨)" in sheet
    assert "네 네 네 네" in sheet


def test_review_jsonl_round_trip(tmp_path):
    """The sidecar contract: one JSON object per line."""
    path = tmp_path / "s1-261.current.review.jsonl"
    rows = [{"start": 0.0, "end": 1.0, "text": "가", "flags": []}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    from review_sheet import load

    assert load(path) == rows


# ─── spelling variants of the same spoken word ───────────────────────────────

ALIASES_TSV = "# canonical<TAB>variants\nACK\t에크|애크|액트|액\nTCP\ttcp\n"


def test_aliases_separate_transliteration_from_misrecognition(tmp_path):
    """26-1 wrote "ACK 메시지", the current pipeline writes "에크 메세지" for the
    same speech. As written that is a difference; as sound it is not."""
    aliases = load_aliases(write(tmp_path, "aliases.tsv", ALIASES_TSV))
    gold = "ACK 메시지를 보내고 ACK 넘버를 확인합니다."
    hyp = "에크 메시지를 보내고 액트 넘버를 확인합니다."
    assert cer(gold, hyp) > 0.05
    assert cer_canonical(gold, hyp, aliases) == pytest.approx(0.0)


def test_canonicalize_prefers_the_longer_variant(tmp_path):
    aliases = load_aliases(write(tmp_path, "aliases.tsv", ALIASES_TSV))
    assert canonicalize("액트 메시지", aliases) == "ACK 메시지"


def test_term_rate_still_counts_an_unusable_spelling_as_wrong(tmp_path):
    """"에크 넘버" is the right sound but no reader recognizes ACK in it."""
    aliases = load_aliases(write(tmp_path, "aliases.tsv", "ACK\t에크\tnotonly\n"))
    entries = load_glossary(write(tmp_path, "terms.tsv", "ACK 넘버\t\t\n"))
    rate, missed, total = term_error_rate("ACK 넘버를 봅니다", "에크 넘버를 봅니다", entries, aliases)
    assert (missed, total) == (1, 1)
    assert rate == pytest.approx(1.0)


def test_term_rate_accepts_a_recognizable_spelling(tmp_path):
    """"힙소트" and "heapsort" are the same term to a reader, so no error."""
    aliases = load_aliases(write(tmp_path, "aliases.tsv", "힙 소트\t힙소트|heapsort\n"))
    entries = load_glossary(write(tmp_path, "terms.tsv", "힙 소트\t\t\n"))
    rate, missed, total = term_error_rate("heapsort를 봅니다", "힙소트를 봅니다", entries, aliases)
    assert (missed, total) == (0, 1)
    assert rate == pytest.approx(0.0)


# ─── nested terms must not double count ──────────────────────────────────────


def test_count_terms_masks_nested_matches():
    counts = count_terms("완전이진트리와 이진트리", {"완전이진트리", "이진트리"})
    assert counts == {"완전이진트리": 1, "이진트리": 1}


def test_nested_terms_do_not_inflate_the_denominator(tmp_path):
    entries = load_glossary(write(tmp_path, "terms.tsv", "ACK\t\t\nACK 넘버\t\t\n"))
    # One "ACK 넘버" is one occurrence, not one of each.
    _, _, total = term_error_rate("ACK 넘버를 확인합니다", "ACK 넘버를 확인합니다", entries)
    assert total == 1


# ─── term replacement guards ─────────────────────────────────────────────────

# Every case here is a real false positive the vault-wide dry run produced
# before the guards were tightened (2026-09-22).
MAPPING = [("세커먼트", "세그먼트"), ("Merge Sort", "머지 소트"), ("애크", "ACK"),
           ("에크", "ACK"), ("액트", "ACK"), ("액", "ACK")]


def fixed(text):
    return replace_guarded(text, MAPPING)[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("에크를 먼저 보내", "ACK를 먼저 보내"),
        ("에크메세지는 뺍니다", "ACK메세지는 뺍니다"),
        ("에크 메세지를 받을 수", "ACK 메세지를 받을 수"),
        ("바로 액을 보낸다", "바로 ACK을 보낸다"),
        ("액트 메시지를 보내", "ACK 메시지를 보내"),
        ("큐에 넣는 세커먼트 하나를", "큐에 넣는 세그먼트 하나를"),
        ("다음에 볼 Merge Sort는", "다음에 볼 머지 소트는"),
    ],
)
def test_replacement_fires_where_it_should(text, expected):
    assert fixed(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "무지티, 흰색, 회색, 에크루색",          # a colour name
        "ACK, 애크놀로지먼트, 확인 응답의 약자",      # part of a longer word
        "Merge Sorting에 대한 방법",              # Latin variant eating a suffix
        "참치액젓만 넣은 파스타",                   # 액 inside another word
        "혈액을 자주 돌리면",                      # same, with a particle after
        "여기 보면 컴프 액트 킬 되어 있습니다",     # compact kill, not ACK
    ],
)
def test_replacement_declines_false_positives(text):
    assert fixed(text) == text


# The particle list decides where a Hangul variant ends. It was missing the
# endings below, which left three confirmed terms unreplaced in the vault
# (2026-09-22): "인터럴트냐", "인터럴트든", "스택 하나리라던지", "인터럴트입니다".
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("이것이 인터럴트냐 하면", "이것이 인터럽트냐 하면"),
        ("인터럴트든 예외든", "인터럽트든 예외든"),
        ("스택 하나리라던지 해서", "스택 카나리라던지 해서"),
        ("핸들러의 인터럴트입니다", "핸들러의 인터럽트입니다"),
    ],
)
def test_replacement_reaches_past_the_added_endings(text, expected):
    mapping = [("인터럴트", "인터럽트"), ("스택 하나리", "스택 카나리")]
    assert replace_guarded(text, mapping)[0] == expected


# Widening the endings must not loosen the short-variant guard: these are the
# false positives the vault-wide dry run produced, now with a copula attached.
@pytest.mark.parametrize("text", ["총액입니다 그리고", "혈액이다 라고", "참치액젓이다"])
def test_copula_endings_do_not_unlock_short_variants(text):
    assert fixed(text) == text


def test_align_keeps_the_word_the_two_generations_spell_differently():
    """The boundary word is the one worth measuring, so it must survive the cut.

    Alignment anchors on runs of MIN_ANCHOR characters that match exactly, and
    the two transcriptions stop agreeing precisely where one of them misheard.
    Cutting at the matched span alone therefore drops the misrecognition — here
    "힙부터 봅니다", against the re-run's "팁부터" — and the metric scores it as a
    deletion instead of the substitution it is.
    """
    cut, _ratio = align(SHIPPED_FULL, RERUN)
    assert "빌드 힙부터 봅니다" in cut
    assert "발췌 밖" not in cut  # widening must not run past the excerpt


# ─── canonicalize boundary guards ────────────────────────────────────────────

# aliases.tsv ships the single syllable 액 as an ACK variant. An unguarded
# str.replace turned 금액을 into 금ACK을 — symmetric between gold and hypothesis,
# so it mostly cancels, but it triples the cost of a real one-character error
# inside such a word.
@pytest.mark.parametrize("text", ["금액을 확인합니다", "혈액 검사 결과", "액체 상태로"])
def test_canonicalize_leaves_ordinary_korean_alone(tmp_path, text):
    aliases = load_aliases(write(tmp_path, "aliases.tsv", ALIASES_TSV))
    assert canonicalize(text, aliases) == text


# The guard must not be so tight that it stops folding the variants the metric
# exists to fold — this is the assertion that catches over-tightening.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("에크 메시지를 보내고", "ACK 메시지를 보내고"),
        ("액트 넘버를 확인", "ACK 넘버를 확인"),
        ("바로 액을 보낸다", "바로 ACK을 보낸다"),
    ],
)
def test_canonicalize_still_folds_real_variants(tmp_path, text, expected):
    aliases = load_aliases(write(tmp_path, "aliases.tsv", ALIASES_TSV))
    assert canonicalize(text, aliases) == expected


# ─── the sidecar audit record ────────────────────────────────────────────────

# The pipeline writes `fixed` without a human in the loop, so it has to say what
# actually happened. It used to infer the list by diffing strings, which both
# invented replacements and dropped real ones.
def test_audit_record_omits_a_variant_a_longer_one_consumed():
    mapping = [("액트", "ACK"), ("액", "ACK")]  # longest first, as load_mapping sorts it
    _text, applied, _declined = replace_guarded("액트 메시지를 받으면", mapping)
    assert [(v, c) for v, c, _at in applied] == [("액트", "ACK")]


def test_audit_record_keeps_the_occurrence_that_fired_when_another_declined():
    # "혈액에크" glues 에크 to more Hangul, so that one is declined; the first is not.
    text = "에크를 받았고 혈액에크 상태"
    fixed, applied, declined = replace_guarded(text, [("에크", "ACK")])
    assert fixed == "ACK를 받았고 혈액에크 상태"
    assert len(applied) == 1 and len(declined) == 1
