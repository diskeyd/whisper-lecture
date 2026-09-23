"""Shared text loading and normalization for every metric in this repo.

Every number has to come from the same normalization, otherwise CER and term
error rate are measured on slightly different strings and stop being comparable.
"""

import re
import unicodedata
from pathlib import Path

# Timestamps written into transcript text: the block header "**0:00 - 1:00**",
# its h:mm:ss form in the older transcripts ("**01:00:00 - 01:01:00**"), and the
# single "**3:20**" marks the correction sheet puts in front of each segment.
# None of them may reach the metrics: the shipped slice keeps absolute times
# (20:00 - 21:00) while a re-run of the same excerpt starts at 0:00, and a gold
# file corrected from the sheet would otherwise carry a few stray digits per
# segment — a few hundred characters of pure noise on a 1,300-character excerpt.
TIMESTAMP_RE = re.compile(
    r"\*\*\d{1,2}:\d{2}(?::\d{2})?(?: - \d{1,2}:\d{2}(?::\d{2})?)?\*\*"
)

# Notes the corrector leaves in a gold file — "<차이들이라고 들리지만 자식일 것 같다>".
# They are commentary, not speech, so they must not be measured as characters.
NOTE_RE = re.compile(r"<[^<>\n]*>")


def load_body(path):
    """Read a transcript .md without timestamp marks or <...> corrector notes."""
    text = Path(path).read_text(encoding="utf-8")
    return NOTE_RE.sub(" ", TIMESTAMP_RE.sub(" ", text))


def normalize(text):
    """NFC, then drop whitespace and punctuation.

    NFC first: macOS/iCloud stores Hangul decomposed (NFD), and comparing NFD
    against NFC counts every syllable as several jamo-level edits.
    Korean spacing is inconsistent between speakers and engines, so it is noise
    for CER; punctuation is the model's invention, not the speaker's.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(
        ch for ch in text if not ch.isspace() and unicodedata.category(ch)[0] != "P"
    )


def eojeol(text):
    """NFC word (eojeol) list — for the omission rate, where spacing must survive."""
    return unicodedata.normalize("NFC", text).split()


# The transcript's one-minute block headers: "**3:00 - 4:00**", and past the hour
# "**01:03:00 - 01:04:00**". A candidate without a timestamp cannot be checked
# against the audio, which is the whole point of reporting it.
HEADER_RE = re.compile(r"^\*\*(\d{1,2}):(\d{2})(?::(\d{2}))? - ")


def blocks(path):
    """→ [(timestamp label, text), ...] — the text of each one-minute block."""
    out, stamp, buf = [], "0:00", []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            if buf:
                out.append((stamp, " ".join(buf)))
            buf = []
            stamp = line.strip().strip("*").split(" - ")[0]
        elif line.strip():
            buf.append(line.strip())
    if buf:
        out.append((stamp, " ".join(buf)))
    return out


# ─── guarded replacement ─────────────────────────────────────────────────────
# One implementation, three callers: fix_terms.py (the vault-wide rewriter),
# transcribe_whisper.py (applied right after a transcript is saved), and
# canonicalize() below. They used to be three copies; the constants drifted
# apart on 2026-09-22 and the pipeline silently stopped fixing three confirmed
# terms that fix_terms fixed.
#
# Short Hangul replacement without boundary rules breaks real Korean: a
# vault-wide dry run produced "참치액젓" → "참치ACK젓" and "혈액을" → "혈ACK을".
HANGUL = re.compile(r"[가-힣]")  # .match for a boundary check, .findall to count syllables
# A Hangul variant is only replaced when what follows marks it as its own word:
# a particle glued to it, or one of the words the term actually travels with.
# Without this, "컴프 액트 킬"(compact kill) became "컴프 ACK 킬".
COMPANIONS = ("메시지", "메세지", "넘버", "번호", "신호", "값", "패킷", "세그먼트", "메쎄지")
PARTICLES = ("을", "를", "이", "가", "은", "는", "도", "만", "에", "의", "로", "으로",
             "와", "과", "랑", "부터", "까지", "에서", "에게", "한테", "이나", "나",
             "라고", "이라고", "처럼", "만큼", "이란", "란", "밖에", "라는", "이라는",
             # 2026-09-22: these left three confirmed terms unreplaced —
             # "인터럴트냐", "인터럴트든", "스택 하나리라던지".
             "냐", "든", "든지", "라던지", "이라던지", "라든지", "이라든지",
             # copula: "인터럴트입니다" was the last one left unreplaced
             "입니다", "이다", "예요", "이에요", "이야", "였", "이었")
# Up to this many Hangul syllables a variant could be the start of an ordinary
# word ("에크"→에크루, "액트"→컴프 액트), so it needs a particle or companion.
# Longer garbled terms ("세커먼트", "인솔차조트") cannot, and only need to not be
# glued to more Hangul.
SHORT_HANGUL = 2


def replace_guarded(text, mapping):
    """Rewrite variants to their canonical spelling, minding Korean word boundaries.

    → (new_text, applied, declined). Each list holds (variant, canonical, offset)
    so a caller that wants to show the surrounding words — or the block timestamp
    they fall under — can build that itself. This module has no business knowing
    about transcript block headers.

    The offset locates the hit in the text as it stood after that variant's pass;
    later variants in the mapping shift it by their own net length change. That
    is a few characters at most, against block headers that sit hundreds of
    characters apart and a display window of ±14, so it is accurate enough for
    both uses and exact enough for neither to be worth a second pass.

    `mapping` is [(variant, canonical), ...]; pass it longest variant first or a
    short variant will consume the start of a long one.
    """
    text = unicodedata.normalize("NFC", text)
    applied, declined = [], []
    for variant, canonical in mapping:
        if not variant or variant == canonical:
            continue
        out, i, shift = [], 0, 0
        while True:
            j = text.find(variant, i)
            if j < 0:
                out.append(text[i:])
                break
            end = j + len(variant)
            before, rest = text[j - 1 : j], text[end:]
            attached = rest.lstrip().startswith(COMPANIONS) or rest.startswith(PARTICLES)
            if variant[-1:].isascii() and variant[-1:].isalnum():
                # Latin variant: "Merge Sort" must not swallow "Merge Sorting"
                risky = rest[:1].isalnum() and rest[:1].isascii()
            elif len(HANGUL.findall(variant)) <= SHORT_HANGUL:
                risky = bool(HANGUL.match(before or " ")) or not attached
            else:
                glued = bool(HANGUL.match(rest[:1] or " ")) and not attached
                risky = bool(HANGUL.match(before or " ")) or glued
            if risky:
                declined.append((variant, canonical, j + shift))
                out.append(text[i:end])
            else:
                applied.append((variant, canonical, j + shift))
                out.append(text[i:j])
                out.append(canonical)
                shift += len(canonical) - len(variant)
            i = end
        text = "".join(out)
    return text, applied, declined


def load_aliases(path):
    """Read glossary/aliases.tsv → [(canonical, [variant, ...], note_usable), ...].

    note_usable=False (third column `notonly`) means the variant is the same
    speech but cannot be used in a note, so the term error rate must still count
    it as wrong even though CER does not.
    """
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        canonical = unicodedata.normalize("NFC", parts[0].strip())
        variants = [
            unicodedata.normalize("NFC", v.strip())
            for v in (parts[1] if len(parts) > 1 else "").split("|")
            if v.strip()
        ]
        note_usable = not (len(parts) > 2 and parts[2].strip() == "notonly")
        if canonical and variants:
            out.append((canonical, variants, note_usable))
    return out


def canonicalize(text, aliases, note_usable_only=False):
    """Rewrite spelling variants of the same spoken word to one form.

    CER uses every alias: "에크 메시지" and "ACK 메시지" are the same speech heard
    right, and the difference between them is not a recognition error.
    The term error rate passes note_usable_only=True, which keeps only the
    variants a reader would still recognize ("힙소트" = "heapsort") and leaves the
    unusable ones ("에크" for ACK) to count as term errors.

    Goes through replace_guarded, not str.replace: aliases.tsv lists the single
    syllable 액 as an ACK variant, so an unguarded pass turned 금액을 into 금ACK을
    and 혈액 into 혈ACK. Symmetric mangling mostly cancels between gold and
    hypothesis, but it amplifies a genuine one-character error inside such a word
    — gold 혈액 against hyp 혈핵 scores three edits instead of one.

    This makes the pass conservative rather than optimistic: a bare 액 with no
    particle after it is now left alone, which can raise the canonical CER. On
    the 2026-09-22 corpus nothing is declined, so no published number moves.
    """
    mapping = [
        (variant, canonical)
        for canonical, variants, note_usable in aliases
        if not (note_usable_only and not note_usable)
        for variant in variants
    ]
    mapping.sort(key=lambda kv: -len(kv[0]))
    return replace_guarded(text, mapping)[0]


def load_glossary(path, course_id=None):
    """Read glossary/terms.tsv → [(correct, [wrong, ...]), ...].

    Same format as transcribe_whisper.py's own loader (tab separated, '#'
    comments, '|' between wrong spellings, optional course id in column 3).
    """
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        correct = unicodedata.normalize("NFC", parts[0].strip())
        if not correct:
            continue
        scope = parts[2].strip() if len(parts) > 2 else ""
        if scope and course_id and scope != course_id:
            continue
        wrong = [
            unicodedata.normalize("NFC", w.strip())
            for w in (parts[1] if len(parts) > 1 else "").split("|")
            if w.strip()
        ]
        entries.append((correct, wrong))
    return entries
