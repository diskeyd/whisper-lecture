"""Judge two transcripts of the same excerpt only where they disagree.

Usage:
  uv run scripts/judge_sheet.py make <stem>     # writes data/judge/<stem>.md
  uv run scripts/judge_sheet.py score <stem> [<stem> ...]

A gold file corrected from one engine's draft keeps that draft's wording
wherever the audio is ambiguous: s1–s3 (whisper drafts) crowned whisper and
s4–s5 (Qwen drafts) crowned Qwen. Listening only where the two transcripts
differ, with both candidates side by side, removes that pull.

`make` aligns data/hyp/<stem>.current.md (whisper) with data/hyp/<stem>.qwen.md
word by word. Spans that differ only in spacing, punctuation or digit-vs-Hangul
numbers are not shown. Times come from the Qwen chunk marks, interpolated by
word position, so they are good to a few seconds.

Fill each `판정:` line with W (whisper is right), Q (Qwen is right), B (both
acceptable) or `X: <what was said>` (neither). Blank lines and `?` count as
unjudged and are left out of every count and of the CER. Free text without a
letter is read as X; heard text that repeats the shown context is trimmed to
the span, and if it then equals one candidate it counts as that candidate.

`score` reports, per stem and in total, how often each engine won a judged
span, and the CER of each transcript against a reference built from the
agreed text plus the judged spans.
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import jiwer
from textnorm import NOTE_RE, load_body, normalize

REPO = Path(__file__).resolve().parent.parent
HYP = REPO / "data" / "hyp"
JUDGE = REPO / "data" / "judge"
MARK_RE = re.compile(r"\*\*(\d+):(\d{2})\*\*")
DIGITS = "영일이삼사오육칠팔구"
CONTEXT = 4  # words of agreed text shown on each side
GAP = 1  # agreed words that may sit inside one item before it is split


def korean_number(match):
    n = int(match.group())
    if n >= 100:
        return match.group()
    tens, ones = divmod(n, 10)
    if not tens:
        return DIGITS[ones]
    return ("" if tens == 1 else DIGITS[tens]) + "십" + (DIGITS[ones] if ones else "")


def key(text):
    """What a span must differ in to be worth listening to."""
    return normalize(re.sub(r"\d+", korean_number, text))


def words(text):
    return [w for w in text.split() if key(w)]


def qwen_words(path):
    """Qwen words with an estimated start time from the **m:ss** chunk marks."""
    raw = Path(path).read_text(encoding="utf-8")
    parts = MARK_RE.split(raw)
    starts = [int(m) * 60 + int(s) for m, s in zip(parts[1::3], parts[2::3])]
    chunks = [words(NOTE_RE.sub(" ", t)) for t in parts[3::3]]
    out = []
    for i, (start, ws) in enumerate(zip(starts, chunks)):
        end = starts[i + 1] if i + 1 < len(starts) else start + 30
        out += [(w, start + (end - start) * j / max(len(ws), 1)) for j, w in enumerate(ws)]
    return out


def items(a, b_timed):
    """Disagreeing spans between word lists a (whisper) and b (Qwen)."""
    b = [w for w, _ in b_timed]
    found = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if key(" ".join(a[i1:i2])) == key(" ".join(b[j1:j2])):
            continue
        if found and i1 - found[-1]["i2"] <= GAP and j1 - found[-1]["j2"] <= GAP:
            found[-1].update(i2=i2, j2=j2)
        else:
            found.append({"i1": i1, "i2": i2, "j1": j1, "j2": j2})
    for f in found:
        f["w"] = " ".join(a[f["i1"]:f["i2"]])
        f["q"] = " ".join(b[f["j1"]:f["j2"]])
        f["before"] = " ".join(b[max(0, f["j1"] - CONTEXT):f["j1"]])
        f["after"] = " ".join(b[f["j2"]:f["j2"] + CONTEXT])
        f["t"] = b_timed[min(f["j1"], len(b_timed) - 1)][1] if b_timed else 0
    return found


def mmss(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def render(stem, found):
    lines = [
        f"# {stem} — 불일치 판정",
        "",
        f"오디오: `data/excerpt/{stem}.excerpt.m4a`. 시각은 ±몇 초. 판정: W · Q · B · `X: 들린 말`.",
        "",
    ]
    for n, f in enumerate(found, 1):
        lines += [
            f"## {n} · {mmss(f['t'])}",
            f"…{f['before']} [ ] {f['after']}…",
            f"- W: {f['w'] or '(없음)'}",
            f"- Q: {f['q'] or '(없음)'}",
            "- 판정: ",
            "",
        ]
    return "\n".join(lines)


def make(stem):
    a = words(load_body(HYP / f"{stem}.current.md"))
    found = items(a, qwen_words(HYP / f"{stem}.qwen.md"))
    JUDGE.mkdir(parents=True, exist_ok=True)
    out = JUDGE / f"{stem}.md"
    if out.exists():
        raise SystemExit(f"{out} exists — judgments would be lost")
    out.write_text(render(stem, found), encoding="utf-8")
    print(f"{out}: {len(found)} spans")


VERDICT_RE = re.compile(r"^- 판정:[ \t]*(.*)$", re.MULTILINE)


def verdicts(text):
    return [v.strip() for v in VERDICT_RE.findall(text)]


SAME = ("똑같", "동일", "상관없")


def trim(heard, before, after):
    """Drop the shown context the listener wrote around the span."""
    h, before, after = key(heard), key(before), key(after)
    for n in range(min(len(h), len(before)), 0, -1):
        if before.endswith(h[:n]):
            h = h[n:]
            break
    for n in range(min(len(h), len(after)), 0, -1):
        if after.startswith(h[-n:]):
            h = h[:-n]
            break
    return h


def verdict(v, f):
    """(W | Q | B | X | "", reference text, X leaning W/Q or None)."""
    head = v[:1].upper()
    if head in ("W", "Q", "B"):
        return head, f["w"] if head == "W" else f["q"], None
    if not v or head == "?":
        return "", "", None
    if any(s in v for s in SAME):
        return "B", f["q"], None
    heard = trim(NOTE_RE.sub(" ", v.split(":", 1)[1] if head == "X" else v), f["before"], f["after"])
    w, q = key(f["w"]), key(f["q"])
    if heard == w == q:
        return "B", f["q"], None
    if heard == w:
        return "W", f["w"], None
    if heard == q:
        return "Q", f["q"], None
    lean = "W" if jiwer.cer(heard, w or "∅") < jiwer.cer(heard, q or "∅") else "Q"
    return "X", heard, lean


def score_one(stem):
    a = words(load_body(HYP / f"{stem}.current.md"))
    b_timed = qwen_words(HYP / f"{stem}.qwen.md")
    found = items(a, b_timed)
    judged = verdicts((JUDGE / f"{stem}.md").read_text(encoding="utf-8"))
    if len(judged) != len(found):
        raise SystemExit(f"{stem}: sheet has {len(judged)} verdicts, alignment has {len(found)}")
    b = [w for w, _ in b_timed]
    tally = {"W": 0, "Q": 0, "B": 0, "X": 0, "": 0, "XW": 0, "XQ": 0}
    ref, hw, hq, i, j = [], [], [], 0, 0
    for f, v in zip(found, judged):
        ref += b[j:f["j1"]]
        hq += b[j:f["j1"]]
        hw += a[i:f["i1"]]
        head, text, lean = verdict(v, f)
        tally[head] += 1
        if lean:
            tally["X" + lean] += 1
        if head:  # unjudged spans (e.g. chatter before class) leave every text
            ref.append(text)
            hw.append(f["w"])
            hq.append(f["q"])
        i, j = f["i2"], f["j2"]
    ref += b[j:]
    hq += b[j:]
    hw += a[i:]
    reference = key(" ".join(ref))
    return tally, jiwer.cer(reference, key(" ".join(hw))), jiwer.cer(reference, key(" ".join(hq)))


def score(stems):
    total = {"W": 0, "Q": 0, "B": 0, "X": 0, "": 0, "XW": 0, "XQ": 0}
    for stem in stems:
        tally, cer_w, cer_q = score_one(stem)
        for k in total:
            total[k] += tally[k]
        print(
            f"{stem:10s} W {tally['W']:3d}  Q {tally['Q']:3d}  둘다맞음 {tally['B']:3d}  "
            f"둘다틀림 {tally['X']:3d} (W쪽 {tally['XW']:2d} Q쪽 {tally['XQ']:2d})  미판정 {tally['']:3d}   CER whisper {cer_w:.3f}  Qwen {cer_q:.3f}"
        )
    decided = total["W"] + total["Q"]
    if decided:
        print(f"{'합계':10s} W {total['W']}  Q {total['Q']}  → whisper 승률 {total['W'] / decided:.0%}")
    near = decided + total["XW"] + total["XQ"]
    if near:
        print(f"{'':10s} 둘다틀림을 가까운 쪽에 더하면 → whisper {(total['W'] + total['XW']) / near:.0%}")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("make", "score"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "make":
        for stem in sys.argv[2:]:
            make(stem)
    else:
        score(sys.argv[2:])


if __name__ == "__main__":
    main()
