"""Find where two recordings of the same lecture were transcribed differently.

Usage:
  uv run scripts/device_diff.py <pair> <label_a> <a.review.jsonl> <label_b> <b.review.jsonl> [<b2.review.jsonl> ...]

Two devices recorded the same room at the same time, so every difference between
their transcripts is a place where at least one of them is wrong — and neither
transcript can say which. The spans are listed for a third opinion (NotebookLM's
own ASR, or the owner listening to a clip of both recordings), largest first.

Side B may be several sidecars in playback order: the iPhone recording of
시스템보안 w4-1(3) was split into two files while the iPad ran through.

Writes data/exp/verify/<pair>/00_목차.md (the checklist) and spans.json (the
same spans with per-device file index and seconds, for cutting clips).
"""

import bisect
import difflib
import json
import sys
from pathlib import Path

from textnorm import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "exp" / "verify"
MIN_CHARS = 8  # shorter differences are one syllable or a particle, not worth a clip
TOP = 15


def load_side(paths):
    """→ (normalized text, [(offset, file_index, start, end, raw text), ...])."""
    text, segs = "", []
    for fi, path in enumerate(paths):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = rec.get("text") or ""
            norm = normalize(raw)
            if not norm:
                continue
            segs.append((len(text), fi, rec["start"], rec["end"], raw.strip()))
            text += norm
    return text, segs


def segment_at(segs, offset):
    """The segment whose normalized text contains this character offset."""
    starts = [s[0] for s in segs]
    i = bisect.bisect_right(starts, offset) - 1
    return segs[max(0, i)]


def diff_spans(text_a, text_b):
    sm = difflib.SequenceMatcher(None, text_a, text_b, autojunk=False)
    spans = [
        (max(i2 - i1, j2 - j1), i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal" and max(i2 - i1, j2 - j1) >= MIN_CHARS
    ]
    spans.sort(key=lambda s: -s[0])
    return spans


def build(label_a, paths_a, label_b, paths_b):
    text_a, segs_a = load_side(paths_a)
    text_b, segs_b = load_side(paths_b)
    rows = []
    for n, (_size, i1, i2, j1, j2) in enumerate(diff_spans(text_a, text_b)[:TOP], 1):
        sa, sb = segment_at(segs_a, i1), segment_at(segs_b, j1)
        rows.append(
            {
                "n": n,
                label_a: {"file": sa[1], "start": sa[2], "end": sa[3], "segment": sa[4],
                          "span": text_a[i1:i2]},
                label_b: {"file": sb[1], "start": sb[2], "end": sb[3], "segment": sb[4],
                          "span": text_b[j1:j2]},
            }
        )
    return rows, len(text_a), len(text_b)


def mmss(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def render(pair, label_a, label_b, rows, len_a, len_b, paths_b):
    lines = [
        f"# {pair} — 두 기기가 다르게 적은 대목",
        "",
        f"{label_a} {len_a}자 · {label_b} {len_b}자 · 8자 이상 차이 상위 {len(rows)}건.",
        "각 건에서 맞는 쪽에 표시. NLM 열은 NotebookLM 이 두 소스에서 인용한 결과.",
        "",
    ]
    for r in rows:
        a, b = r[label_a], r[label_b]
        b_file = f" (파일 {b['file'] + 1})" if len(paths_b) > 1 else ""
        lines += [
            f"## {r['n']:02d}  {label_a} {mmss(a['start'])} · {label_b} {mmss(b['start'])}{b_file}",
            "",
            f"- **{label_a}**: {a['segment']}",
            f"- **{label_b}**: {b['segment']}",
            f"- 차이: 「{a['span'] or '—'}」 ↔ 「{b['span'] or '—'}」",
            "- NLM: ",
            f"- [ ] {label_a} 맞음  - [ ] {label_b} 맞음  - [ ] 둘 다 틀림",
            "",
        ]
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    if len(args) < 5:
        print(__doc__)
        return 1
    pair, label_a, path_a, label_b, *paths_b = args
    rows, len_a, len_b = build(label_a, [path_a], label_b, paths_b)
    out = OUT / pair
    out.mkdir(parents=True, exist_ok=True)
    (out / "00_목차.md").write_text(
        render(pair, label_a, label_b, rows, len_a, len_b, paths_b), encoding="utf-8"
    )
    (out / "spans.json").write_text(
        json.dumps({"a": [path_a], "b": paths_b, "labels": [label_a, label_b], "rows": rows},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"{out / '00_목차.md'}: {len(rows)}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
