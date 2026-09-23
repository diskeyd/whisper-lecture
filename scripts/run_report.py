"""Summarize data/run-log.jsonl — what each transcription run actually cost.

Usage:
  uv run scripts/run_report.py [YYYY-MM-DD]

transcribe_whisper.py appends one line per file. GPU utilisation itself is not
recorded: powermetrics needs sudo, so it cannot run unattended. What is recorded
is wall time per stage, and mlx-whisper saturates the GPU while it runs — so
transcribe + recheck seconds are the GPU-busy time, and upload seconds are not.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "data" / "run-log.jsonl"


def load(day=None):
    if not LOG.exists():
        return []
    rows = [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if not day or r.get("at", "").startswith(day)]


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else None)
    if not rows:
        print(f"기록 없음: {LOG}")
        return 0
    by_day = defaultdict(list)
    for r in rows:
        by_day[r.get("at", "?")[:10]].append(r)

    for day, group in sorted(by_day.items()):
        audio = sum(r.get("audio_min") or 0 for r in group)
        tr = sum(r.get("transcribe_s") or 0 for r in group)
        rc = sum(r.get("recheck_s") or 0 for r in group)
        up = sum(r.get("upload_s") or 0 for r in group)
        mb = sum(r.get("source_mb") or 0 for r in group)
        n_rc = sum(r.get("recheck_n") or 0 for r in group)
        print(f"\n## {day} — 파일 {len(group)}개 · 오디오 {audio / 60:.1f}시간 · 원본 {mb / 1000:.2f} GB")
        print(f"  GPU   전사 {tr / 60:5.1f}분" + (f" + 재확인 {rc / 60:.1f}분({n_rc}건)" if rc else "")
              + f"  = {(tr + rc) / 60:.1f}분")
        print(f"  네트워크 업로드 {up / 60:5.1f}분")
        print(f"  합계 {(tr + rc + up) / 60:5.1f}분 (오디오 1시간당 {(tr + rc + up) / max(audio, 1) * 60 / 60:.1f}분)")
        print(f"\n  {'파일':38} {'분':>5} {'MB':>6} {'전사s':>6} {'배속':>5} {'재확인':>6} {'업로드s':>7}")
        for r in sorted(group, key=lambda x: -(x.get("transcribe_s") or 0)):
            rcs = f"{r.get('recheck_s', 0):.0f}s/{r.get('recheck_n', 0)}" if r.get("recheck_s") else "—"
            print(f"  {r['file'][:38]:38} {r.get('audio_min') or 0:5.1f} {r.get('source_mb') or 0:6.1f}"
                  f" {r.get('transcribe_s') or 0:6.0f} {r.get('speed_x') or 0:5.1f} {rcs:>6}"
                  f" {r.get('upload_s') or 0:7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
