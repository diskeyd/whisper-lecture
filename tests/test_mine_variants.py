"""Guards that keep the mining report readable.

Both regressions come from the first run over the vault's security-course notes
(2026-09-22): with ~180 acronym-heavy terms in scope, window fragments buried the
real finds — "IP → IPS" alone claimed 281 hits, and it is not a misspelling of
anything, it is the term IP standing on its own.
"""

import mine_variants as mv


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(f"**0:00 - 1:00**\n{text}\n", encoding="utf-8")
    return p


def mine_one(tmp_path, text, terms):
    path = write(tmp_path, "t.md", text)
    return {cand: term for term, cand in mv.mine([path], [(t, []) for t in terms])}


def test_latin_fragment_is_not_a_misspelling(tmp_path):
    """IP/ID/LS are fragments of IPS/IDS/TLS, not what Whisper wrote instead."""
    found = mine_one(tmp_path, "IP 주소를 보고 LS 계층에서 ID 를 확인해요", ["IPS", "TLS", "IDS"])
    assert found == {}


def test_term_prefix_candidate_is_skipped(tmp_path):
    """Saying the short form is shorthand, not a mis-transcription of the long form."""
    found = mine_one(tmp_path, "이진트리 형태로 만들어요", ["완전 이진트리"])
    assert found == {}


def test_real_hangul_misspelling_still_found(tmp_path):
    """The guards must not silence the case the tool exists for."""
    found = mine_one(tmp_path, "베스트 호스트를 앞에 둡니다", ["베스천 호스트"])
    assert found == {"베스트 호스트": "베스천 호스트"}


def test_long_latin_misspelling_still_found(tmp_path):
    """Four characters or more stays in scope — only 2-3 char windows are dropped."""
    found = mine_one(tmp_path, "DNSSEK 을 켜야 돼요", ["DNSSEC"])
    assert found == {"DNSSEK": "DNSSEC"}


def test_sibling_compound_is_not_a_misspelling(tmp_path):
    """'부모 프로세스' pulled in 자식/멀티/싱글/특정/모든 프로세스 — ten rows of nothing."""
    found = mine_one(tmp_path, "자식 프로세스를 만들고 멀티 프로세스로 돌려요", ["부모 프로세스"])
    assert found == {}


def test_garbled_word_in_a_compound_is_still_found(tmp_path):
    """The guard judges the differing word: 부모 → 분모 is a garble, 부모 → 자식 is not."""
    found = mine_one(tmp_path, "분모 노드를 먼저 봅니다", ["부모 노드"])
    assert found == {"분모 노드": "부모 노드"}


# 2026-09-22: the loop stopped at the first term that scored in band, so the
# reported pair depended on set iteration order — the same input gave
# 프로로그→프로그램 on five runs out of six and 프로로그→프롤로그 on the other.
def test_the_best_scoring_term_wins_not_the_first_one(tmp_path):
    # "ready 큐" scores 76.9 against ready키, "ready 상태" only 71.4. A human had
    # to correct this pair by hand; scoring every term gets it right on its own.
    got = mine_one(tmp_path, "idle키가 아니라 ready키로 넘어갑니다", ["ready 큐", "ready 상태"])
    assert got["ready키"] == "ready 큐"


def test_a_doubled_letter_inside_an_acronym_stays_unminable(tmp_path):
    """BBFT→BFT cannot be mined and it is worth knowing why.

    `term in cand` treats a term found inside the candidate as an inflection —
    that is what makes "ACK를" read as ACK rather than a misspelling of it. "BFT"
    is a substring of "BBFT", so the right answer is dropped before scoring and
    only PBFT remains. Loosening the rule floods the report, so this class of
    error has to come from reading the transcript, not from mining it.
    """
    assert mine_one(tmp_path, "PBFT, BBFT 계열 중에", ["PBFT", "BFT"]) == {"BBFT": "PBFT"}


def test_a_real_tie_is_reported_as_a_tie(tmp_path):
    """Two-syllable Korean scores only 0/50/100, so ties are common and similarity
    cannot break them. Naming both leaves the judgement to whoever reads the
    report; picking one silently is how a wrong pair gets into the glossary."""
    got = mine_one(tmp_path, "함수의 프로로그 코드입니다", ["프로그램", "프롤로그"])
    assert got["프로로그"] == "프로그램 | 프롤로그"


def test_since_uses_the_filename_date_not_mtime(tmp_path):
    from datetime import date

    names = ["260923 알고리즘w4-1.md", "260909 알고리즘w2-1.md", "월간Obsidian 10회차.md",
             "261340 bad date.md"]
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_text("x", encoding="utf-8")
        paths.append(p)
    got = mv.recent(paths, 7, today=date(2026, 9, 23))
    assert [p.name for p in got] == ["260923 알고리즘w4-1.md"]
