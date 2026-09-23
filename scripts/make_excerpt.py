"""Cut a 5-minute excerpt out of a raw recording.

Usage:
  uv run scripts/make_excerpt.py <stem> <start_minute>

The start must be a whole minute: the shipped transcripts are one-minute blocks
counted from 0, so a minute-aligned cut maps to exactly blocks [m, m+5) and
slice_transcript.py can pull the same minutes out of the .md.

-c copy snaps to a keyframe, so the result is verified against 300 s and
re-encoded only if the copy landed too far off.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXCERPT = ROOT / "data" / "excerpt"
MANIFEST = ROOT / "data" / "manifest.json"
LENGTH = 300
TOLERANCE = 2.0


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def cut(src, dst, start, reencode=False, length=LENGTH):
    codec = ["-c:a", "aac", "-b:a", "128k"] if reencode else ["-c", "copy"]
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-t", str(length),
         "-i", str(src), *codec, "-y", str(dst)],
        check=True,
    )


def record(stem, info):
    data = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    data.setdefault(stem, {}).update(info)
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    stem, start_min = sys.argv[1], int(sys.argv[2])

    src = next(iter(sorted(RAW.glob(stem + ".*"))), None)
    if src is None:
        raise SystemExit(f"no raw file for stem '{stem}' in {RAW}")
    EXCERPT.mkdir(parents=True, exist_ok=True)
    dst = EXCERPT / f"{stem}.excerpt{src.suffix}"

    src_duration = duration(src)
    start = start_min * 60
    if start + LENGTH > src_duration:
        raise SystemExit(
            f"start {start_min} min + 5 min exceeds the recording ({src_duration / 60:.1f} min)"
        )

    cut(src, dst, start)
    got = duration(dst)
    if abs(got - LENGTH) > TOLERANCE:
        print(f"⚠  stream copy gave {got:.1f}s — re-encoding for an exact cut")
        cut(src, dst, start, reencode=True)
        got = duration(dst)
    print(f"{dst.name}: {got:.1f}s from {start_min}:00")

    record(stem, {
        "start_minute": start_min,
        "excerpt": dst.name,
        "excerpt_seconds": round(got, 1),
        "raw_seconds": round(src_duration, 1),
        "raw_megabytes": round(src.stat().st_size / 1e6, 1),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
