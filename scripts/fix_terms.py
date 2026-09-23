"""Rewrite broken term spellings in transcripts that are already written.

Usage:
  uv run scripts/fix_terms.py <file.md|dir> ...          # dry run, reports only
  uv run scripts/fix_terms.py --write <file.md|dir> ...  # actually edit the files

Why this and not a better model: the only terminology tool the paid APIs
document as *guaranteed* is deterministic replacement (AWS `DisplayAs`,
Deepgram `replace`, AssemblyAI `custom_spelling`) — every one of them needs the
mapping enumerated by hand. The same mapping works here for nothing, and it
applies to the transcripts that already exist without re-transcribing anything.

Sources of the mapping:
  glossary/aliases.tsv  — variants of the same spoken word (에크 → ACK)
  glossary/terms.tsv    — the wrong-spelling column (세커먼트 → 세그먼트)

Guarding, learned from a dry run over all 208 vault transcripts (2026-09-22),
which produced these false positives before the rules were tightened:
"에크루색" → ACK루색, "애크놀로지먼트" → ACK놀로지먼트, "Merge Sorting" →
머지 소트ing, and Snort rules `alert tcp any` → `alert TCP any`.

  * Korean glues particles straight onto a word ("에크를", "tct라고"), so
    "followed by Hangul" cannot disqualify a match. But Hangul that continues
    into another word must: a match is only replaced when what follows is a
    particle, one of the words the term travels with, or nothing Hangul at all.
  * A Latin variant must not be followed by another letter or digit, or
    "Merge Sort" eats "Merge Sorting".
  * A one-syllable variant ("액") needs a particle or companion — a bare "액"
    between spaces is more likely to be something else.

Everything it declines to touch is reported, not hidden. Run the dry run and
read it before ever passing --write.
"""

import re
import sys
from pathlib import Path

from textnorm import load_aliases, load_glossary, replace_guarded

REPO = Path(__file__).resolve().parent.parent
# Block headers, so a reported change can be checked against the audio.
HEADER_RE = re.compile(r"^\*\*(\d{1,2}:\d{2}(?::\d{2})?) - ", re.MULTILINE)


def load_mapping():
    """→ [(variant, canonical), ...], longest variant first."""
    pairs = {}
    for canonical, variants, _usable in load_aliases(REPO / "glossary" / "aliases.tsv"):
        for v in variants:
            pairs.setdefault(v, canonical)
    for correct, wrong in load_glossary(REPO / "glossary" / "terms.tsv"):
        for w in wrong:
            pairs.setdefault(w, correct)
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def stamp_at(text, index):
    """The block timestamp that covers this character offset ("" if none)."""
    last = ""
    for m in HEADER_RE.finditer(text):
        if m.start() > index:
            break
        last = m.group(1)
    return last


def context_at(text, index, width=14):
    """The words around a hit, prefixed with its block timestamp when there is one.

    `replace_guarded` reports offsets, not strings — it has no business knowing
    about transcript block headers — so the display context is built here.
    """
    snippet = text[max(0, index - width) : index + width * 2].replace("\n", " ")
    stamp = stamp_at(text, index)
    return f"{stamp} …{snippet}…" if stamp else snippet


def targets(args):
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from sorted(p.rglob("*.md"))
        else:
            yield p


def main():
    args = [a for a in sys.argv[1:] if a != "--write"]
    write = "--write" in sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    mapping = load_mapping()
    print(f"{len(mapping)} 개 표기 규칙" + ("" if write else "  (모의 실행 — 파일을 바꾸지 않습니다)"))

    totals, declined_totals, changed = {}, {}, 0
    for path in targets(args):
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"⚠  {path}: {e}")
            continue
        new, applied, declined = replace_guarded(original, mapping)
        for v, c, at in applied:
            totals.setdefault((v, c), []).append((path.name, context_at(original, at)))
        for v, c, at in declined:
            declined_totals.setdefault((v, c), []).append((path.name, context_at(original, at)))
        if new != original:
            changed += 1
            if write:
                path.write_text(new, encoding="utf-8")

    print(f"\n바꿀 파일 {changed} 개")
    if totals:
        print("\n적용:")
        for (v, c), hits in sorted(totals.items(), key=lambda kv: -len(kv[1])):
            print(f"  {v} → {c}  {len(hits)}건")
            for name, ctx in hits[:2]:
                print(f"      {name}  {ctx}")
    if declined_totals:
        print("\n건너뜀 (다른 낱말에 붙어 있거나 짧은 표기):")
        for (v, c), hits in sorted(declined_totals.items(), key=lambda kv: -len(kv[1])):
            print(f"  {v} → {c}  {len(hits)}건 보류")
            for name, ctx in hits[:2]:
                print(f"      {name}  {ctx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
