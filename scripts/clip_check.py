"""Check that each verification clip really holds the word it was cut for, and recut the ones that do not.

Usage (brew python3.14 — mlx-whisper lives there, not in the repo venv nor the default python3 3.15):
  python3.14 scripts/clip_check.py check <clip dir> [work dir]
  python3.14 scripts/clip_check.py recut <clip dir> [work dir]

The clips were cut at the one-minute block stamps of the shipped vault
transcripts, and those clocks drift — up to 3 min 17 s over a 48 min lecture
(measured 2026-09-22). So a clip can miss its word entirely:
01_BBFT_38m00s was cut at audio 38:00 while "BBFT" is spoken at 41:17.

check   transcribes every clip, aligns it into the vault transcript by content,
        and reports whether the candidate spelling falls inside that stretch.
        The block the clip landed in gives that file's drift for free.
recut   downloads the originals of the failures from Drive, re-transcribes a
        window around candidate_stamp + drift, and cuts a new clip on the time
        whisper reports for the segment — an audio clock, not the vault's.

Nothing is written into the repo: clip transcripts and downloads go to the work
dir (a scratchpad), and only the clips and the two index files in the clip dir
are replaced.
"""

import difflib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # transcribe_whisper.py
from fetch import gws
from make_excerpt import cut
from slice_transcript import MIN_RATIO, covered, find_in_vault, locate
from textnorm import blocks, normalize

MODEL = "mlx-community/whisper-large-v3-mlx"
LANGUAGE = "ko"

CLIP_LENGTH = 68  # what the existing clips are, kept so the owner's estimate holds
LEAD = 8  # seconds of run-up before the word, so it is not the first thing heard
EDGE_MARGIN = 15  # normalized chars (~3 s): nearer a boundary than this is a risky cut
MAX_SPREAD = 3.0  # a cut longer than this × the clip text is a stray match, not an alignment
WINDOW_PAD = 120  # first search window either side of the estimated time
WIDE_PAD = 360  # second try before giving up on a window and taking the whole file
CONTEXT_SPAN = 200  # normalized chars of vault text either side of a candidate

TOC = "00_목차.md"
ORDER = "00_듣는순서.md"
RESULTS = "clips.tsv"
AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".mp4"}

HEAD_RE = re.compile(r"^##\s*(\d+)\.\s*`(.+?)`")
HEAR_RE = re.compile(r"^-\s*들을 것:\s*\*\*(.+?)\s*→\s*(.+?)\*\*")
SRC_RE = re.compile(r"^-\s*원본:\s*(.+?)\s*\(([^()]+?)\s*구간\)\s*$")
NAME_RE = re.compile(r"^(\d+)_(.+)_(\d+)m(\d+)s\.m4a$")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def stamp_seconds(label):
    """'38:00', '02:07:00' or '127:00' → seconds."""
    parts = [int(p) for p in label.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def hhmm(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def parse_toc(path):
    """00_목차.md → (entries, skipped).

    Each entry is {num, clip, cands: [(cand, target)], source, stamp}. Read from
    the index, not from the filenames: clips 04 and 31 carry two candidates each,
    and a filename can only hold one.

    An entry without a usable `- 원본:` line goes to `skipped` instead of taking
    the run down with it. The index is hand-edited, and a whole pass costs an
    hour of decoding; one malformed line should not spend it.
    """
    out, skipped, cur = [], [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = HEAD_RE.match(line)
        if m:
            cur = {"num": m.group(1), "clip": nfc(m.group(2)), "cands": [],
                   "source": None, "stamp": None}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = HEAR_RE.match(line)
        if m:
            cur["cands"].append((nfc(m.group(1).strip()), nfc(m.group(2).strip())))
            continue
        m = SRC_RE.match(line)
        if m:
            cur["source"] = nfc(m.group(1).strip())
            try:
                cur["stamp"] = stamp_seconds(m.group(2))
            except ValueError:
                cur["stamp"] = None
    good = []
    for e in out:
        if e["source"] and e["stamp"] is not None and e["cands"]:
            good.append(e)
        else:
            skipped.append((e["num"], e["clip"]))
    return good, skipped


def haystack(path):
    """A vault transcript as one normalized string + the block stamp of every character."""
    hay, stamps = [], []
    for label, text in blocks(path):
        n = normalize(text)
        hay.append(n)
        stamps.extend([stamp_seconds(label)] * len(n))
    return "".join(hay), stamps


def judge(hay, lo, hi, cand):
    """Is the candidate inside the stretch the clip covers, with room on both sides?

    → (status, position). A word sitting on the boundary is reported separately:
    the clip technically contains it but starts or stops mid-phrase.
    """
    want = normalize(cand)
    pos = hay.find(want, lo, hi)
    if pos < 0:
        return "miss", None
    if pos - lo < EDGE_MARGIN or hi - (pos + len(want)) < EDGE_MARGIN:
        return "edge", pos
    return "ok", pos


def transcribe(path):
    import mlx_whisper  # type: ignore[import-not-found]  # brew python3.14 only; the repo venv has no mlx

    return mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=MODEL,
        verbose=False,
        language=LANGUAGE,
        condition_on_previous_text=False,
    )


def clip_text(clip, work):
    """Transcribe a clip once and keep it, so a re-run costs nothing."""
    cached = work / f"{Path(clip).stem}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    text = transcribe(clip)["text"].strip()
    cached.write_text(text + "\n", encoding="utf-8")
    return text


def vault_path(source):
    from transcribe_whisper import COURSE_MAP, detect_course  # needs mlx at import

    key = detect_course(source)
    if key is None:
        raise SystemExit(f"no course keyword in {source}")
    return find_in_vault(COURSE_MAP[key], source)


# ─── check ────────────────────────────────────────────────────────────────────


def cmd_check(clip_dir, work):
    entries, skipped = parse_toc(clip_dir / TOC)
    for num, clip in skipped:
        print(f"{num}  ⚠ 목차 항목을 읽을 수 없어 건너뜁니다: {clip}")
    vaults, rows = {}, []
    for e in entries:
        clip = clip_dir / e["clip"]
        if not clip.exists():
            print(f"{e['num']}  missing clip: {e['clip']}")
            continue
        text = clip_text(clip, work)
        if e["source"] not in vaults:
            vaults[e["source"]] = haystack(vault_path(e["source"]))
        hay, stamps = vaults[e["source"]]

        found = covered(hay, normalize(text))
        spread = (found[1] - found[0]) / max(1, len(normalize(text))) if found else 0
        if found is None or found[2] < MIN_RATIO or spread > MAX_SPREAD:
            for cand, target in e["cands"]:
                rows.append({**base(e, cand, target), "status": "noalign",
                             "ratio": round(found[2], 2) if found else 0.0,
                             "landed": "", "drift": ""})
            print(f"{e['num']}  ✗ 정렬 실패  {e['clip']}")
            continue

        lo, hi, ratio = found
        drift = e["stamp"] - stamps[lo]  # + = the audio runs ahead of the vault clock
        for cand, target in e["cands"]:
            status, _ = judge(hay, lo, hi, cand)
            rows.append({**base(e, cand, target), "status": status, "ratio": round(ratio, 2),
                         "landed": hhmm(stamps[lo]), "drift": drift})
            mark = {"ok": "✅", "edge": "⚠", "miss": "❌"}[status]
            print(f"{e['num']}  {mark} {cand:<16} 블록 {hhmm(stamps[lo])}  "
                  f"드리프트 {drift:+d}s  ratio {ratio:.2f}")

    write_rows(work / RESULTS, rows)
    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    print(f"→ {work / RESULTS}")
    return 0


def base(e, cand, target):
    return {"num": e["num"], "clip": e["clip"], "cand": cand, "target": target,
            "source": e["source"], "stamp": e["stamp"]}


FIELDS = ["num", "clip", "cand", "target", "source", "stamp", "status", "ratio",
          "landed", "drift"]


def write_rows(path, rows):
    lines = ["\t".join(FIELDS)]
    lines += ["\t".join(str(r[f]) for f in FIELDS) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [dict(zip(FIELDS, line.split("\t"))) for line in lines[1:] if line.strip()]


# ─── recut ────────────────────────────────────────────────────────────────────


def drive_candidates(date):
    """Every recording on Drive whose name carries this yymmdd.

    Searched by name rather than inside DRIVE_FOLDER_ID: that folder holds the
    current semester only, and four of these clips come from the spring term,
    which sits in a different folder.
    """
    params = json.dumps({"q": f"name contains '{date}' and trashed=false",
                         "fields": "files(id,name,size)", "pageSize": 100})
    out = gws(["drive", "files", "list", "--params", params, "--format", "json"])
    return [f for f in json.loads(out).get("files", [])
            if Path(nfc(f["name"])).suffix.lower() in AUDIO_SUFFIXES]


def drive_match(files, source):
    """Drive's name for a transcript's recording.

    The two do not always agree: the transcript 260902 시스템보안w1-2(2) is stored
    on Drive as 260902 시스템보안(2), with the week dropped. So fall back from the
    exact stem to date prefix + course keyword + the trailing (n).
    """
    from transcribe_whisper import detect_course

    stem = nfc(Path(source).stem)
    for f in files:
        if nfc(Path(f["name"]).stem) == stem:
            return f
    date, course = stem[:6], detect_course(stem)
    tail = re.search(r"\((\d+)\)\s*$", stem)
    tail = tail.group(1) if tail else None
    hits = []
    for f in files:
        s = nfc(Path(f["name"]).stem)
        t = re.search(r"\((\d+)\)\s*$", s)
        if (s.startswith(date) and detect_course(s) == course
                and (t.group(1) if t else None) == tail):
            hits.append(f)
    return hits[0] if len(hits) == 1 else None


def download(meta, dest):
    if dest.exists():
        return dest
    params = json.dumps({"fileId": meta["id"], "alt": "media"})
    gws(["drive", "files", "get", "--params", params, "-o", dest.name], cwd=dest.parent)
    return dest


def window_segments(src, start, length, work):
    """Re-encode the window to 16k mono before decoding.

    -c copy would snap the start to a keyframe and every absolute time computed
    as window_start + segment.start would inherit that error.
    """
    wav = work / "window.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-t", str(length),
                    "-i", str(src), "-ac", "1", "-ar", "16000", "-y", str(wav)], check=True)
    return transcribe(wav)["segments"]


def vault_context(hay, stamps, stamp, cand, span=CONTEXT_SPAN):
    """The vault text around the occurrence of cand nearest the given block stamp."""
    want = normalize(cand)
    best, at = -1, 0
    while True:
        p = hay.find(want, at)
        if p < 0:
            break
        if best < 0 or abs(stamps[p] - stamp) < abs(stamps[best] - stamp):
            best = p
        at = p + 1
    if best < 0:
        return ""
    return hay[max(0, best - span): best + len(want) + span]


def segment_time(segments, cand, context, near=None, span=CONTEXT_SPAN):
    """Seconds into the window where cand is spoken, or None.

    Direct match first: the same model on the same audio usually reproduces its
    own misrecognition. The vault context covers the run that spells it
    differently this time.

    A word worth checking is one the lecturer repeats, so a window usually holds
    several occurrences and the right one has to be picked, not the first —
    펀테이너 sits in the 3:00 and the 4:00 block, and taking the first gave both
    clips the same 3:52. Timing cannot separate them: the drift is derived from
    block labels a minute wide and the occurrences are a minute apart. So the
    context decides whenever there is one, even against a single direct match —
    a repeated word is exactly the one the re-run spells right in one place and
    wrong in another, and then the only match in the window is the wrong one.
    `near` breaks ties only when there is no context at all.
    """
    parts, owner = [], []
    for i, s in enumerate(segments):
        n = normalize(s["text"])
        parts.append(n)
        owner.extend([i] * len(n))
    hay = "".join(parts)
    want = normalize(cand)
    hits, at = [], 0
    while True:
        p = hay.find(want, at)
        if p < 0:
            break
        hits.append(p)
        at = p + 1

    # locate(), not covered(): this is an anchor, not a measurement. The matched
    # span is where the two texts demonstrably agree, and the fraction below
    # interpolates inside it. Widening by the unmatched ends would stretch the
    # span by text that did not match and move the interpolated point.
    found = locate(hay, context) if context else None
    if found and found[2] >= MIN_RATIO:
        lo, hi = found[0], found[1]
        inside = [p for p in hits if lo <= p < hi]
        if inside:
            hits = inside
        else:
            # Spelled differently here, so anchor on where the word sits inside
            # the stretch the context aligned to.
            frac = max(0, context.find(want)) / max(1, len(context))
            hits = [min(len(owner) - 1, lo + int(frac * (hi - lo)))]
    if not hits:
        return None
    if len(hits) > 1:
        if context:
            hits.sort(key=lambda p: -difflib.SequenceMatcher(
                None, hay[max(0, p - span): p + len(want) + span], context).ratio())
        elif near is not None:
            hits.sort(key=lambda p: abs(segments[owner[p]]["start"] - near))
    return segments[owner[hits[0]]]["start"]


def find_time(src, stamp, drift, cand, context, work):
    """Absolute seconds in the recording where cand is spoken.

    The drift is not uniform inside a file (one measured transcript jumps from
    block 04:00 to 30:00), so widen twice before falling back to the whole file.
    """
    guess = stamp + drift
    for pad in (WINDOW_PAD, WIDE_PAD):
        start = max(0, guess - pad)
        segs = window_segments(src, start, 2 * pad, work)
        t = segment_time(segs, cand, context, near=guess - start)
        if t is not None:
            return start + t
    segs = transcribe(src)["segments"]
    return segment_time(segs, cand, context, near=guess)


def renamed(old, t):
    m = NAME_RE.match(old)
    if not m:
        return old
    return f"{m.group(1)}_{m.group(2)}_{int(t) // 60:02d}m{int(t) % 60:02d}s.m4a"


def plan_recuts(rows):
    """Which clips to recut, and which to leave for a human → (recut, manual).

    Grouped by clip, not by candidate: clips 04 and 31 carry two candidates each.
    Recutting is per clip, so a clip whose other candidate is fine cannot be recut
    without losing it — the new cut is centred on the failing candidate and the
    old file is deleted. A mixed clip is therefore reported, not touched.

    → recut is [(clip, [row, ...])] holding the rows that drove the decision; the
    first one is what the new cut is centred on, matching the word already in the
    filename. manual is the same shape, with every row of the mixed clip.
    """
    bad_states = ("miss", "edge", "noalign")
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)
    recut, manual = [], []
    for clip, group in by_clip.items():
        bad = [r for r in group if r["status"] in bad_states]
        if not bad:
            continue
        (manual if len(bad) < len(group) else recut).append((clip, group if bad != group else bad))
    return recut, manual


def cmd_recut(clip_dir, work):
    rows = read_rows(work / RESULTS)
    recut_plan, manual = plan_recuts(rows)
    for clip, group in manual:
        mixed = ", ".join(f"{r['cand']}({r['status']})" for r in group)
        print(f"   ⚠ 수동 확인  {clip} — 후보가 섞여 있어 건드리지 않습니다: {mixed}")
    bad = [group[0] for _clip, group in recut_plan]
    if not bad:
        print("재절단할 클립이 없습니다.")
        return 0

    vaults, renames, report, listings = {}, {}, [], {}
    # A file's other clips measured its drift; a clip that failed to align has none of its own.
    drifts = {}
    for r in rows:
        if r["drift"] != "":
            drifts.setdefault(r["source"], []).append(int(r["drift"]))

    for r in bad:
        source = r["source"]
        if source not in vaults:
            vaults[source] = haystack(vault_path(source))
        hay, stamps = vaults[source]
        date = nfc(Path(source).stem)[:6]
        if date not in listings:
            listings[date] = drive_candidates(date)
        meta = drive_match(listings[date], source)
        if meta is None:
            report.append((r, None, "Drive 에서 원본을 못 찾음"))
            print(f"{r['num']}  ✗ Drive 미발견  {source}")
            continue
        # Keyed by Drive id, not by clip: two clips often share one recording,
        # and the id is stable, ASCII, and cannot route the file into the vault.
        src = download(meta, work / f"{meta['id']}{Path(nfc(meta['name'])).suffix}")

        stamp = int(r["stamp"])
        drift = int(r["drift"]) if r["drift"] != "" else round(
            sum(drifts.get(source, [0])) / max(1, len(drifts.get(source, [0]))))
        context = vault_context(hay, stamps, stamp, r["cand"])
        t = find_time(src, stamp, drift, r["cand"], context, work)
        if t is None:
            report.append((r, None, "원본에서 해당 구간을 못 찾음"))
            print(f"{r['num']}  ✗ 구간 미발견  {r['cand']}")
            continue

        new = renamed(r["clip"], t)
        cut(src, clip_dir / new, max(0, t - LEAD), reencode=True, length=CLIP_LENGTH)
        if new != r["clip"]:
            renames[r["clip"]] = new
            (clip_dir / r["clip"]).unlink(missing_ok=True)
        report.append((r, t, "재생성"))
        print(f"{r['num']}  ✅ {r['cand']:<16} {hhmm(stamp)} → {hhmm(t)}  ({new})")

    patch_docs(clip_dir, renames)
    write_rows(work / "recut.tsv",
               [{**r, "status": note, "landed": hhmm(t) if t else "", "drift": r["drift"]}
                for r, t, note in report])
    print(f"\n재생성 {sum(1 for _, t, _ in report if t)} / 실패 "
          f"{sum(1 for _, t, _ in report if not t)}")
    return 0


def patch_docs(clip_dir, renames):
    """Keep the two index files pointing at the clips that exist.

    A plain filename swap, not a regeneration: 00_듣는순서.md carries the
    hand-written reason each term has to be listened to.
    """
    if not renames:
        return
    for name in (TOC, ORDER):
        path = clip_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in renames.items():
            # old is already NFC (parse_toc normalizes), so the second pass has to
            # be the decomposed form — iCloud hands filenames back as NFD and the
            # index may have been generated from such a listing.
            text = text.replace(old, new).replace(unicodedata.normalize("NFD", old), new)
        path.write_text(text, encoding="utf-8")
    print(f"목차·듣는순서 파일명 {len(renames)}건 갱신")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("check", "recut"):
        print(__doc__)
        return 1
    cmd, clip_dir = sys.argv[1], Path(sys.argv[2]).expanduser()
    work = Path(sys.argv[3]).expanduser() if len(sys.argv) > 3 else clip_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    return cmd_check(clip_dir, work) if cmd == "check" else cmd_recut(clip_dir, work)


if __name__ == "__main__":
    sys.exit(main())
