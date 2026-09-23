"""Cut the excerpt's span out of a shipped transcript in the satellite vault.

Usage:
  uv run scripts/slice_transcript.py <stem> <course_id> <transcript filename>

  uv run scripts/slice_transcript.py s1-261 "201.66 컴퓨터네트워크" \
      "260512 컴퓨터네트워크w11-1.md"

Gives data/hyp/<stem>.shipped.md — the same stretch of lecture as the audio
excerpt, as it was shipped before the 2026-08-26 config change. Read-only on
the vault.

Why this aligns by content instead of by timestamp (measured 2026-09-22):
the old transcripts' clocks drift. In 260512 컴퓨터네트워크w11-1 the audio at
48:00 lands under the header "**46:00 - 47:00**" — roughly two minutes lost
over 48. Cutting the nominal minutes would compare two different stretches of
lecture and every CER number would be noise. So the shipped text is aligned
against the re-run of the same excerpt (data/hyp/<stem>.current.md), which is
audio-accurate, and cut at the matching boundaries. The re-run is used only to
locate the span, never as a reference.
"""

import difflib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import load_body, normalize

VAULT = Path(
    "<WIKI_VAULT>"
)
SOURCES = VAULT / "10. Raw Sources"
HYP = Path(__file__).resolve().parent.parent / "data" / "hyp"
MIN_ANCHOR = 12  # a matching run shorter than this is coincidence, not alignment
MIN_RATIO = 0.35  # below this the two texts are not the same stretch of lecture


def nfc(s):
    return unicodedata.normalize("NFC", s)


def find_in_vault(course_id, filename):
    """iCloud stores Hangul as NFD, so match on normalized listings, not on glob."""
    for course_dir in SOURCES.iterdir():
        if course_dir.is_dir() and nfc(course_dir.name) == nfc(course_id):
            for entry in course_dir.iterdir():
                if nfc(entry.name) == nfc(filename):
                    return entry
            raise SystemExit(f"transcript not found: {course_id}/{filename}")
    raise SystemExit(f"course folder not found: {course_id}")


def normalize_with_map(text):
    """Normalized text plus, for each kept character, its index in `text`.

    `text` must already be NFC: composing per character cannot merge a base
    syllable with the combining jamo that follows it, so decomposed input has to
    be normalized as a whole first (the caller does it).
    """
    keep, index = [], []
    for i, ch in enumerate(text):
        if normalize(ch):
            keep.append(ch)
            index.append(i)
    return "".join(keep), index


def locate(hay, needle):
    """Where in hay the needle's speech sits, by the outermost long matching runs.

    → (lo, hi, ratio, head, tail), or None when nothing matches for MIN_ANCHOR
    characters. head and tail are how much of the needle falls outside [lo, hi)
    at each end; the caller decides whether to widen by them.

    Both callers need the raw span too: recut anchors a search window on the
    matched text only, where widening would blur the anchor.
    """
    runs = [b for b in difflib.SequenceMatcher(None, hay, needle).get_matching_blocks()
            if b.size >= MIN_ANCHOR]
    if not runs:
        return None
    lo, hi = runs[0].a, runs[-1].a + runs[-1].size
    ratio = difflib.SequenceMatcher(None, hay[lo:hi], needle).ratio()
    return lo, hi, ratio, runs[0].b, len(needle) - (runs[-1].b + runs[-1].size)


def covered(hay, needle):
    """The stretch of hay the needle covers, widened by its unmatched ends.

    → (lo, hi, ratio), or None.

    Alignment is built on runs of MIN_ANCHOR characters that match exactly, and
    two transcriptions of the same audio stop agreeing that closely precisely
    where one of them misheard a word. So the matched span alone ends just short
    of the misrecognition at each edge — the very thing being measured — and the
    metric scores those characters as deletions instead of substitutions.
    Measured on s1-261 (2026-09-22): head 11, tail 27, CER 0.1958 → 0.1860.

    Widening by the needle's unmatched length is an estimate: the two texts
    differ in length there too. Using the sub-MIN_ANCHOR matching blocks as
    secondary anchors was tried and is worse — coincidental 1-2 character
    matches scatter across a 48-minute transcript — and constraining them to
    the window leaves CER identical to four decimal places.
    """
    found = locate(hay, needle)
    if found is None:
        return None
    lo, hi, ratio, head, tail = found
    return max(0, lo - head), min(len(hay), hi + tail), ratio


def align(shipped_text, anchor_text):
    """The stretch of shipped_text that covers the same speech as anchor_text.

    → (cut_text, ratio).
    """
    shipped_text = unicodedata.normalize("NFC", shipped_text)
    hay, index = normalize_with_map(shipped_text)
    found = covered(hay, normalize(anchor_text))
    if found is None:
        raise SystemExit("no alignment found between the shipped transcript and the re-run")
    lo, hi, ratio = found
    return shipped_text[index[lo] : index[hi - 1] + 1], ratio


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    stem, course_id, filename = sys.argv[1], sys.argv[2], sys.argv[3]

    anchor = HYP / f"{stem}.current.md"
    if not anchor.exists():
        raise SystemExit(f"{anchor} is missing — transcribe the excerpt first")

    src = find_in_vault(course_id, filename)
    shipped = load_body(src)  # headers out: their clock is what drifted
    cut, ratio = align(shipped, load_body(anchor))
    if ratio < MIN_RATIO:
        raise SystemExit(
            f"alignment ratio {ratio:.2f} is too low — wrong file, or the excerpt is not "
            "in this transcript"
        )

    body = cut.strip() + "\n"
    HYP.mkdir(parents=True, exist_ok=True)
    out = HYP / f"{stem}.shipped.md"
    out.write_text(body, encoding="utf-8")

    manifest = HYP.parent / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data.setdefault(stem, {}).update(
            {"shipped_source": nfc(src.name), "shipped_align_ratio": round(ratio, 3)}
        )
        manifest.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"{out.name}: {len(normalize(body))} chars from {nfc(src.name)} (align {ratio:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
