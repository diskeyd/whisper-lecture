"""Harvest candidate misspellings of known terms out of the transcripts we already have.

Usage:
  uv run scripts/mine_variants.py <dir|file> ... [--course "201.66 컴퓨터네트워크"] [--since 14]

--since N keeps only transcripts whose YYMMDD filename prefix is within the last N
days — the recurring check mines what arrived since the last round, not the whole
folder again. The filename date is used, not mtime: iCloud sync and fix_terms.py
both touch mtime.

The wrong-spelling column of glossary/terms.tsv cannot be guessed from documents —
it is whatever Whisper actually wrote. But the 200+ transcripts already on disk
*are* the record of what it wrote, so the candidates can be mined instead of
recalled: for every glossary term, look for strings that are close to it but not
equal, and report them with counts and context for the owner to confirm.

Nothing is written to the glossary. Confirmed lines go in by hand — a wrong entry
there makes fix_terms.py corrupt real Korean.
"""

import re
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path

from rapidfuzz import fuzz
from textnorm import blocks, load_glossary

REPO = Path(__file__).resolve().parent.parent
LOW, HIGH = 70, 97  # similarity band: below is a different word, at/above is a match
MIN_HITS = 1
# "ACK를", "시퀀스 넘버가" are inflections, not misspellings — strip the particle and
# drop anything that still contains the term, or the report is all grammar.
TRAILING = ("이라고", "라고", "으로", "에서", "에게", "한테", "까지", "부터", "이나", "라는",
            "이라는", "만큼", "처럼", "밖에", "이란", "을", "를", "이", "가", "은", "는",
            "도", "만", "에", "의", "로", "와", "과", "랑", "나", "란")
# "그 ACK", "이제 윈도우", "뭐 윈도우" — a filler in front of a term is not a
# misspelling either, and these flooded the first run.
FILLERS = ("그", "저", "요", "이", "어", "음", "자", "뭐", "이제", "인제", "지금", "좀",
           "아", "그냥", "막", "딱", "요거", "이거", "저거", "그거")
# A two-letter Latin window is a fragment of a longer acronym ("IP" out of "IPS",
# "LS" out of "TLS"), not something Whisper wrote instead of it — the real failure
# mode for an English acronym in a Korean lecture is Hangul ("에크" for ACK), not a
# shorter acronym. Without this the report drowns: "IP → IPS" alone scored 281 hits.
MIN_LATIN = 4
# A compound term drags in every phrase that shares its head noun: "부모 프로세스"
# scored 71 against 자식/멀티/싱글/특정/모든/단일 프로세스 — ten rows of nothing, because
# two words out of two matching leaves the score high no matter what the other word is.
# So when a whole word is shared, judge the words that differ: a misspelling garbles
# them (부모 → 분모, 67), a different compound replaces them outright (부모 → 자식, 0).
DIFF_FLOOR = 50
# When two terms tie on similarity the report names both rather than picking one.
# Two-syllable Korean scores only 0/50/100, so ties are common and similarity
# genuinely cannot break them: 프로로그 scores 75.0 against both 프로그램 and
# 프롤로그, and only the surrounding sentence says it is the latter. Silently
# choosing would slip a plausible wrong pair into the report — exactly the
# failure a human had to catch by hand on 2026-09-22.
TIE_SEPARATOR = " | "


def _differs_plausibly(cand, term):
    """False when candidate and term share a word and the rest is a clean swap.

    "자식 프로세스" vs "부모 프로세스" is two different compounds, not one written
    wrong. Only what is left after removing the shared words decides.
    """
    a, b = cand.split(), term.split()
    shared = set(a) & set(b)
    if not shared:
        return True
    rest_a = " ".join(w for w in a if w not in shared)
    rest_b = " ".join(w for w in b if w not in shared)
    if not rest_a or not rest_b:  # one is the other plus a word — shorthand, not error
        return False
    return fuzz.ratio(rest_a, rest_b) >= DIFF_FLOOR


def recent(paths, days, today=None):
    """Transcripts whose YYMMDD name prefix falls within the last `days` days."""
    today = today or datetime.now().astimezone().date()
    out = []
    for p in paths:
        m = re.match(r"(\d{2})(\d{2})(\d{2})", unicodedata.normalize("NFC", p.name))
        if not m:
            continue
        try:
            d = date(2000 + int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            continue
        if (today - d).days <= days:
            out.append(p)
    return out


def windows(text):
    """Eojeol n-grams (1–3 words) — a term is rarely longer than three words."""
    words = text.split()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            yield " ".join(words[i : i + n])


def mine(paths, entries):
    known = {unicodedata.normalize("NFC", c) for c, _ in entries}
    found = {}
    for path in paths:
        try:
            chunks = blocks(path)
        except OSError:
            continue
        for stamp, raw in chunks:
            text = unicodedata.normalize("NFC", raw)
            for cand in set(windows(text)):
                bare = cand.rstrip(" .,!?·…\"'()[]{}~-—:;")
                for suffix in TRAILING:  # longest first, one particle is enough
                    if len(bare) > len(suffix) + 1 and bare.endswith(suffix):
                        bare = bare[: -len(suffix)].strip()  # "DNSSEK 을" leaves a space
                        break
                if bare in known or len(bare) < 2:
                    continue
                if bare.isascii() and len(bare) < MIN_LATIN:
                    continue
                head = bare.split(" ", 1)
                if len(head) == 2 and head[0] in FILLERS:
                    continue
                scored = []
                for term in known:
                    if term in cand:  # the term is there, just inflected
                        continue
                    if bare in term:  # a piece of the term, not a rewriting of it
                        continue
                    if abs(len(cand) - len(term)) > max(2, len(term) // 3):
                        continue
                    if not _differs_plausibly(bare, term):
                        continue
                    score = fuzz.ratio(bare, term)
                    if LOW <= score < HIGH:
                        scored.append((score, term))
                if not scored:
                    continue
                # Score every term and take the best. Stopping at the first one
                # that scored in band made the reported pair depend on set
                # iteration order: the same input gave 프로로그→프로그램 on five
                # runs out of six and 프로로그→프롤로그 on the other. The best
                # score is often the right answer too — it picks "ready 큐" over
                # "ready 상태" (76.9 vs 71.4), one of the three pairs a human had
                # to correct by hand on 2026-09-22.
                #
                # It cannot rescue BBFT→BFT, though: `term in cand` above drops
                # BFT before it is ever scored, because "BFT" is a substring of
                # "BBFT". That rule is there so "ACK를" reads as an inflection of
                # ACK rather than a misspelling, and it cannot tell a particle
                # from a doubled letter. Loosening it floods the report, so the
                # limitation stands — a doubled or dropped character inside an
                # acronym is not minable and has to come from reading.
                best = max(s for s, _ in scored)
                tied = sorted(t for s, t in scored if s == best)
                slot = found.setdefault(
                    (TIE_SEPARATOR.join(tied), bare), {"hits": 0, "score": best, "where": []}
                )
                slot["hits"] += text.count(cand)
                slot["where"].append((path.name, stamp))
    return found


def main():
    args = sys.argv[1:]
    course = None
    if "--course" in args:
        i = args.index("--course")
        course = args[i + 1]
        args = args[:i] + args[i + 2 :]
    since = None
    if "--since" in args:
        i = args.index("--since")
        since = int(args[i + 1])
        args = args[:i] + args[i + 2 :]
    if not args:
        print(__doc__)
        return 1

    paths = []
    for a in args:
        p = Path(a)
        paths.extend(sorted(p.rglob("*.md")) if p.is_dir() else [p])
    if since is not None:
        paths = recent(paths, since)
    entries = load_glossary(REPO / "glossary" / "terms.tsv", course)
    print(f"전사문 {len(paths)}개 · 용어 {len(entries)}개" + (f" · 과목 {course}" if course else ""))

    found = mine(paths, entries)
    rows = sorted(found.items(), key=lambda kv: (-kv[1]["hits"], -kv[1]["score"]))
    print(f"\n후보 {len(rows)}건 — 오디오의 해당 시각을 듣고 확인한 뒤 terms.tsv 오답 열에 넣으세요\n")
    for (term, cand), info in rows:
        if info["hits"] < MIN_HITS:
            continue
        print(f"{cand}  →  {term}   (유사도 {info['score']:.0f} · {info['hits']}건)")
        for name, stamp in info["where"][:4]:
            print(f"    {stamp}  {name}")
        if len(info["where"]) > 4:
            print(f"    … 외 {len(info['where']) - 4}곳")
    return 0


if __name__ == "__main__":
    sys.exit(main())
