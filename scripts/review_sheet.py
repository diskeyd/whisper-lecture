"""Turn a .review.jsonl sidecar into a markdown correction sheet.

Usage:
  uv run scripts/review_sheet.py <stem>.current.review.jsonl

Flagged segments first — those are where listening pays off — then the full
draft, so the corrector can work top to bottom and save the result as
data/gold/<stem>.md.

Flags come from transcribe_whisper.py: low_logprob · maybe_silence · repetitive
· term_miss · disagree · recheck_failed. avg_logprob and no_speech_prob are per
30-second decoding window, so neighbouring segments share one value, and a
confident misrecognition carries no flag at all — the full draft still has to be
read.
"""

import json
import sys
from pathlib import Path

FLAG_NOTE = {
    "low_logprob": "낮은 확신",
    "maybe_silence": "무음 가능",
    "repetitive": "반복 필터가 손댐",
    "term_miss": "용어 오인식 후보",
    "disagree": "2차 디코딩 불일치",
    "recheck_failed": "2차 디코딩 실패",
    "spelling_fixed": "표기 자동 치환됨",
}


def two(value):
    return "–" if value is None else f"{value:.2f}"


def mmss(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def load(path):
    records = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠  line {n} is not valid JSON, skipped: {e}")
    return records


def render(records, stem):
    flagged = [r for r in records if r.get("flags")]
    lines = [
        f"# {stem} — 교정 시트",
        "",
        f"세그먼트 {len(records)} · 플래그 {len(flagged)}",
        "",
        "교정 방법: 오디오를 들으며 **틀린 곳만** 고쳐 `data/gold/<stem>.md` 로 저장합니다.",
        "깨진 기술 용어는 `glossary/terms.tsv` 의 오답 열에 실제 표기를 그대로 적습니다.",
        "플래그가 없어도 틀릴 수 있으니(확신에 찬 오인식) 아래 전체 초안도 한 번 읽습니다.",
        "",
        "## 플래그 구간",
        "",
    ]
    if not flagged:
        lines += ["플래그 없음.", ""]
    for r in flagged:
        note = " · ".join(FLAG_NOTE.get(f) or str(f) for f in r["flags"])
        lines.append(f"### {mmss(r['start'])} – {mmss(r['end'])} · {note}")
        lines.append("")
        lines.append(f"- 전사: {r.get('text') or '(삭제됨)'}")
        if r.get("raw_text"):
            lines.append(f"- 필터 이전: {r['raw_text']}")
        if r.get("alt") is not None:
            lines.append(f"- 2차: {r['alt'] or '(빈 결과)'}")
        if r.get("term"):
            lines.append(f"- 용어 사전: {r['term']}")
        if r.get("fixed"):
            changed = " · ".join(f"{v} → {c}" for v, c in r["fixed"])
            lines.append(f"- 자동 치환: {changed}")
        lp, ns = r.get("avg_logprob"), r.get("no_speech_prob")
        if lp is not None or ns is not None:
            fmt = lambda v: "–" if v is None else f"{v:.2f}"
            lines.append(f"- avg_logprob {fmt(lp)} · no_speech {fmt(ns)}")
        lines.append("")

    lines += ["## 전체 초안", ""]
    for r in records:
        text = r.get("text")
        if not text:
            continue
        lines.append(f"**{mmss(r['start'])}** {text}")
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    stem = path.name.split(".")[0]
    records = load(path)
    if not records:
        raise SystemExit(f"no records in {path}")
    out = path.with_name(f"{stem}.sheet.md")
    out.write_text(render(records, stem), encoding="utf-8")
    flagged = sum(1 for r in records if r.get("flags"))
    print(f"{out}: {len(records)} segments, {flagged} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
