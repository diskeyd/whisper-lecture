"""Download one lecture recording from Drive into data/raw/ under a safe stem.

Usage:
  uv run scripts/fetch.py <stem> <folder_id> <drive filename>

The stem must not contain any course keyword from transcribe_whisper.py's
COURSE_MAP: a matching filename routes that script's output into the vault and
would overwrite the shipped transcript we are trying to measure. Checked here.

gws refuses an --output path outside its working directory, so the call runs
with cwd=data/raw and a bare filename.
"""

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"

# Mirrors COURSE_MAP in ../transcribe_whisper.py (checked 2026-09-22).
COURSE_KEYWORDS = [
    "네트워크보안", "데이터통신", "시스템보안", "컴퓨터네트워크", "컴퓨터구조",
    "데이터베이스", "정보보호", "컴퓨터과학개론", "보안기술최신동향", "산학프로젝트",
    "운영체제", "암호학", "알고리즘", "arvr", "신기술산업체", "산업체현장실습",
    "오픈소스플랫폼", "오픈소스프랫폼", "월간obsidian",
]


def gws(args, cwd=None):
    r = subprocess.run(
        ["gws", *args], cwd=cwd, capture_output=True, text=True, timeout=1800, check=False
    )
    if r.returncode != 0:
        raise RuntimeError(f"gws failed: {r.stderr.strip()[:300]}")
    return r.stdout


def nfc(s):
    return unicodedata.normalize("NFC", s)


def find_file(folder_id, name):
    """Locate a file by name inside one folder. Names come back NFD or NFC."""
    params = json.dumps(
        {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "files(id,name,size)",
            "pageSize": 1000,
        }
    )
    files = json.loads(gws(["drive", "files", "list", "--params", params, "--format", "json"]))
    wanted = nfc(name)
    for f in files.get("files", []):
        if nfc(f["name"]) == wanted:
            return f
    raise SystemExit(f"not found in {folder_id}: {name}")


def check_stem(stem):
    low = nfc(stem).lower()
    for kw in COURSE_KEYWORDS:
        if kw.lower() in low:
            raise SystemExit(
                f"stem '{stem}' contains the course keyword '{kw}' — output would be written "
                "into the vault over a shipped transcript. Pick a neutral stem."
            )


def record(stem, info):
    data = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    data.setdefault(stem, {}).update(info)
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def partial_path(out):
    """Where a download lives until it is complete.

    Deliberately not `<stem>.<ext>.part`: make_excerpt.py finds a recording with
    `RAW.glob(stem + ".*")`, so a leftover .part would match and it would cut an
    excerpt out of a truncated file. A leading dot keeps it out of that glob.
    """
    return out.with_name(f".{out.name}.download")


def download(meta, out):
    """Fetch to a temp name and move it into place only once it is whole.

    gws raises on a non-zero exit but leaves what it already wrote, and Ctrl-C
    does the same. The caller then sees the file exists, prints "already
    downloaded", and every later run silently measures a truncated recording.
    """
    tmp = partial_path(out)
    tmp.unlink(missing_ok=True)
    params = json.dumps({"fileId": meta["id"], "alt": "media"})
    try:
        gws(["drive", "files", "get", "--params", params, "-o", tmp.name], cwd=RAW)
        want = int(meta.get("size") or 0)
        got = tmp.stat().st_size
        if want and got != want:
            raise RuntimeError(f"불완전한 다운로드: {got} / {want} 바이트")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    return out


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    stem, folder_id, name = sys.argv[1], sys.argv[2], sys.argv[3]
    check_stem(stem)
    RAW.mkdir(parents=True, exist_ok=True)

    meta = find_file(folder_id, name)
    out = RAW / (stem + Path(nfc(name)).suffix)
    if out.exists():
        print(f"already downloaded: {out}")
    else:
        download(meta, out)
        print(f"downloaded {out} ({out.stat().st_size / 1e6:.1f} MB)")

    record(stem, {"drive_id": meta["id"], "original_name": nfc(meta["name"]), "raw": out.name})
    return 0


if __name__ == "__main__":
    sys.exit(main())
