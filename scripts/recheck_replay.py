"""Re-run only the recheck stage of transcribe_whisper.py under candidate settings.

Usage:
  python3.14 scripts/recheck_replay.py <audio> <sidecar.review.jsonl> <course_id> <setting> ...
  python3.14 scripts/recheck_replay.py --summary <sidecar.review.jsonl> ...

Runs under brew python3.14 (mlx-whisper), like clip_check.py. The pipeline
file is not edited: recheck_flagged() calls mlx_whisper.transcribe through the
module, so the call can be wrapped and its keyword arguments overridden. The
flags from the original run are kept so every setting rechecks the same
segments; only `alt` and the disagree/recheck_failed flags are stripped first.

Output: data/review/<audio stem>.<setting>.review.jsonl — see transcribe_whisper.sidecar_path.
"""

import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from textnorm import normalize

LADDER = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SETTINGS = {
    "A": {},  # as shipped: temperature=0.2, glossary initial_prompt
    "B": {"temperature": LADDER},
    "B2": {"temperature": LADDER[1:]},
    "C": {"temperature": LADDER, "initial_prompt": None},
    "D": {"temperature": LADDER, "initial_prompt": None, "no_speech_threshold": None},
}
DISAGREE_RATIO = 0.6
LOOP_RE = re.compile(r"(.{2,20}?)(?:,?\s*\1){3,}")


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(recs):
    alts = [r for r in recs if r.get("alt") is not None]
    ratios = [similarity(r["alt"], r["text"]) for r in alts]
    control = [r for r in alts if set(r["flags"]) - {"disagree", "recheck_failed"} == {"maybe_silence"}]
    hist = [0] * 11
    for x in ratios:
        hist[min(10, int(x * 10))] += 1
    return {
        "rechecked": sum(1 for r in recs if "disagree" in r["flags"] or "recheck_failed" in r["flags"] or r.get("alt") is not None),
        "alts": len(alts),
        "exact_disagree": sum(1 for r in alts if normalize(r["alt"]) != normalize(r["text"])),
        "ratio_disagree": sum(1 for x in ratios if x < DISAGREE_RATIO),
        "loops": sum(1 for r in alts if LOOP_RE.search(r["alt"])),
        "short": sum(1 for r in alts if len(normalize(r["alt"])) < 0.8 * len(normalize(r["text"]))),
        "prefix_이는": sum(1 for r in alts if r["alt"].startswith(("이는", "이 "))),
        "ends_습니다": sum(1 for r in alts if r["alt"].rstrip(" .").endswith("습니다")),
        "empty": sum(1 for r in alts if not r["alt"]),
        "control_n": len(control),
        "control_disagree": sum(1 for r in control if similarity(r["alt"], r["text"]) < DISAGREE_RATIO),
        "hist_0.1": hist,
    }


def strip(recs):
    for r in recs:
        r["alt"] = None
        r["flags"] = [f for f in r["flags"] if f not in ("disagree", "recheck_failed")]
    return recs


def run(audio, sidecar, course_id, setting):
    import transcribe_whisper as tw

    override = SETTINGS[setting]
    real = tw.mlx_whisper.transcribe
    tw.mlx_whisper.transcribe = lambda a, **kw: real(a, **{**kw, **override})
    try:
        recs = strip(load(sidecar))
        entries = tw.load_glossary(course_id)
        n = tw.recheck_flagged(Path(audio), recs, entries, "ko")
        pairs = tw.load_spelling_map(course_id)
        replace = tw._replace_guarded()
        for r in recs:
            if r.get("alt"):
                r["alt"] = replace(r["alt"], pairs)[0]
    finally:
        tw.mlx_whisper.transcribe = real
    out = tw.write_review_jsonl(Path(audio).with_name(f"{Path(audio).stem}.{setting}.md"), recs)
    print(f"[{setting}] 불일치(원 기준) {n} → {out}")
    return recs


def main():
    args = sys.argv[1:]
    if args and args[0] == "--summary":
        for p in args[1:]:
            print(Path(p).name, json.dumps(summarize(load(p)), ensure_ascii=False))
        return 0
    if len(args) < 4:
        print(__doc__)
        return 1
    audio, sidecar, course_id, *settings = args
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", audio],
        capture_output=True, text=True, check=True).stdout)
    last = load(sidecar)[-1]["end"]
    print(f"오디오 {dur:.0f}s · 사이드카 마지막 세그먼트 {last:.0f}s")
    if abs(dur - last) > 30:
        raise SystemExit("사이드카와 오디오가 다른 파일입니다")
    for s in settings:
        recs = run(audio, sidecar, course_id, s)
        print(f"[{s}]", json.dumps(summarize(recs), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
