"""Term error rate and omission rate for one transcript against the gold text.

Usage:
  uv run scripts/term_error.py <gold.md> <hyp.md> [course_id]

Term error rate: of every glossary term occurrence in the gold text, the share
the hypothesis does not reproduce. A 6% CER can still break every key term,
which is the actual complaint, so this is a first-class number, not a footnote.

Omission rate: share of gold eojeol with no counterpart in the hypothesis —
the fast-speech drops where a few words vanish entirely.
"""

import sys
from pathlib import Path

import jiwer
from textnorm import canonicalize, eojeol, load_aliases, load_body, load_glossary, normalize

GLOSSARY = Path(__file__).resolve().parent.parent / "glossary" / "terms.tsv"
ALIASES = Path(__file__).resolve().parent.parent / "glossary" / "aliases.tsv"


def count_terms(text, needles):
    """Occurrences of each needle, counting every character at most once.

    Longest first with the matched span masked off, so "ACK 넘버" does not also
    count as an "ACK", and "완전 이진트리" does not also count as "이진트리".
    Plain `str.count` per term inflates the denominator and makes the error rate
    look better than it is.
    """
    counts = dict.fromkeys(needles, 0)
    taken = [False] * len(text)
    for needle in sorted(needles, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(needle, start)
            if i < 0:
                break
            if not any(taken[i : i + len(needle)]):
                for j in range(i, i + len(needle)):
                    taken[j] = True
                counts[needle] += 1
            start = i + 1
    return counts


def term_error_rate(gold_text, hyp_text, entries, aliases=()):
    """→ (rate, missed, total). total counts occurrences, not distinct terms.

    Spellings a reader still recognizes are folded together first ("힙소트" =
    "heapsort"); the ones marked `notonly` in aliases.tsv are not, so writing
    "에크" where the term is ACK still counts as wrong.
    """

    def prep(text):
        return normalize(canonicalize(text, aliases, note_usable_only=True))

    needles = {prep(correct) for correct, _wrong in entries if prep(correct)}
    gold_counts = count_terms(prep(gold_text), needles)
    hyp_counts = count_terms(prep(hyp_text), needles)
    missed = total = 0
    for needle, n_gold in gold_counts.items():
        if not n_gold:
            continue  # a term the lecture never said cannot be got wrong
        total += n_gold
        missed += max(0, n_gold - hyp_counts[needle])
    rate = missed / total if total else 0.0
    return rate, missed, total


def omission_rate(gold_text, hyp_text):
    """→ (rate, deletions, gold_eojeol_count).

    ponytail: word-level alignment, so a gold eojeol the hypothesis merged with
    its neighbour ("스택 카나리" → "스택카나리") counts as an omission. Korean
    spacing is unstable; treat this number as an upper bound and read it next to
    CER, not alone.
    """
    gold, hyp = eojeol(gold_text), eojeol(hyp_text)
    if not gold:
        return 0.0, 0, 0
    out = jiwer.process_words(" ".join(gold), " ".join(hyp))
    return out.deletions / len(gold), out.deletions, len(gold)


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 1
    gold_path, hyp_path = sys.argv[1], sys.argv[2]
    course_id = sys.argv[3] if len(sys.argv) == 4 else None

    gold, hyp = load_body(gold_path), load_body(hyp_path)
    entries = load_glossary(GLOSSARY, course_id)
    aliases = load_aliases(ALIASES)
    if not entries:
        print(f"⚠  no glossary entries in {GLOSSARY} (course={course_id}) — term rate skipped")

    rate, missed, total = term_error_rate(gold, hyp, entries, aliases)
    om_rate, deletions, n_eojeol = omission_rate(gold, hyp)
    print(f"term error  {rate:.3f}  ({missed}/{total} occurrences, {len(entries)} terms in scope)")
    print(f"omission    {om_rate:.3f}  ({deletions}/{n_eojeol} eojeol)")
    print(f"gold chars  {len(normalize(gold))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
