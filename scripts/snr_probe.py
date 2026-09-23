"""Rank recordings by how far speech sits above the room's noise floor.

Usage:
  uv run scripts/snr_probe.py <file.m4a|dir> ...

Decodes 60 s at the 25 %, 50 % and 75 % points of each file to 16 kHz mono PCM,
takes the RMS of every 50 ms frame, and reports p90 (speech peaks) minus p10
(room tone) in dB — three windows, then their median. A recording from the back
of the room has a small gap; a close mic has a large one.

This is loudness contrast only. It does not measure reverberation, and a
noise-gated recording (floor at digital silence) scores a meaningless 70+ dB —
the 2026-09 candidate table excluded one such file. The same numbers picked
s3-clean and compared the iPhone/iPad pairs; see results/2026-09-baseline.md.
"""

import array
import math
import statistics
import subprocess
import sys
from pathlib import Path

FRAME = 800  # 50 ms at 16 kHz
WINDOW = 60
POINTS = (0.25, 0.5, 0.75)


def pcm(path, start, seconds=WINDOW):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-t", str(seconds),
         "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    a = array.array("h")
    a.frombytes(out[: len(out) // 2 * 2])
    return a


def duration(path):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout)


def dbfs(rms):
    return 20 * math.log10(rms / 32768) if rms > 0 else -120.0


def gap(path, start):
    """(speech − floor, speech, floor) in dBFS for one window."""
    samples = pcm(path, start)
    frames = sorted(
        math.sqrt(sum(s * s for s in samples[i : i + FRAME]) / FRAME)
        for i in range(0, len(samples) - FRAME, FRAME)
    )
    floor = dbfs(frames[len(frames) // 10])
    speech = dbfs(frames[len(frames) * 9 // 10])
    return speech - floor, speech, floor


def probe(path):
    d = duration(path)
    gaps = [gap(path, d * p) for p in POINTS]
    return statistics.median(g[0] for g in gaps), [g[0] for g in gaps], gaps[1]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    paths = []
    for a in sys.argv[1:]:
        p = Path(a)
        paths.extend(sorted(p.glob("*.m4a")) if p.is_dir() else [p])
    rows = sorted((probe(p), p.stem) for p in paths)
    print(f"{'median':>7} {'25%':>6} {'50%':>6} {'75%':>6}  {'speech/floor@50%':>17}  file")
    for (med, three, (_, speech, floor)), stem in rows:
        print(f"{med:7.1f} {three[0]:6.1f} {three[1]:6.1f} {three[2]:6.1f}  "
              f"{speech:8.1f}/{floor:7.1f}   {stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
