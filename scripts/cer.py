"""CER between two transcripts.

Usage:
  uv run scripts/cer.py <gold.md> <hyp.md>

Prints two numbers:
  적힌 대로   — the text as written. This is the one that matters for the notes.
  표기 통일   — after rewriting spelling variants of the same spoken word
                (glossary/aliases.tsv), so "에크 메시지" counts as "ACK 메시지".
The gap between them is how much of the error is a transliteration choice rather
than a misrecognition.
"""

import sys
from pathlib import Path

import jiwer
from textnorm import canonicalize, load_aliases, load_body, normalize

ALIASES = Path(__file__).resolve().parent.parent / "glossary" / "aliases.tsv"


def cer(gold_text, hyp_text):
    return jiwer.cer(normalize(gold_text), normalize(hyp_text))


def cer_canonical(gold_text, hyp_text, aliases):
    return cer(canonicalize(gold_text, aliases), canonicalize(hyp_text, aliases))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    gold_path, hyp_path = sys.argv[1], sys.argv[2]
    gold, hyp = load_body(gold_path), load_body(hyp_path)
    aliases = load_aliases(ALIASES)

    print(f"{gold_path} vs {hyp_path}")
    print(f"  CER 적힌 대로  {cer(gold, hyp):.3f}")
    print(f"  CER 표기 통일  {cer_canonical(gold, hyp, aliases):.3f}")
    print(f"  정답본 {len(normalize(gold))}자 · 가설 {len(normalize(hyp))}자")
    return 0


if __name__ == "__main__":
    sys.exit(main())
