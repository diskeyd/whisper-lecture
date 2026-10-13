"""Fix misrecognitions left after term replacement, with an LLM reading the text only.

Usage:
  uv run scripts/post_edit.py <transcript.md> ["<course id>"] [--model opus]

Writes <stem>.post.md next to the input and <stem>.post.tsv with every proposed
edit (block, from, to, reason, applied?). The input is never touched.

Conditions from the 2026-09-22 trial (results/2026-09-baseline.md 「레버 ⑥」):
the original stays, every change is listed with its block time so it can be
checked against the audio, it runs after term replacement, and the model
leaves anything it is unsure of alone.

The model does not rewrite the text. It returns edits as exact substrings of
one block, and an edit is applied only when its `from` occurs exactly once in
that block — so a response can fix words but cannot rephrase, summarise or
drop sentences. The `claude` CLI is called with every tool disabled, from a
temp directory, so no project files or instructions reach it.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from textnorm import load_glossary

REPO = Path(__file__).resolve().parent.parent
HEADER_RE = re.compile(r"^\*\*(\d{1,2}:\d{2}(?::\d{2})?) - [^*]+\*\*\s*$")
CHUNK = 10  # blocks per request (≈10 minutes)

RULES = """You correct Korean university lecture transcripts produced by speech recognition.
Fix only clear misrecognitions: a word that is phonetically garbled into a different
word or a non-word, where the surrounding text makes the intended word certain
(technical terms, numbers that follow an obvious sequence, common words).

Do NOT: polish style, fix spacing or punctuation, translate, change spoken fillers or
informal speech, add or remove content, merge or split sentences, or touch the
**m:ss - m:ss** headers. If you are not sure what was said, leave it.

Course: {course}
Correct spellings of terms used in this course (the transcript may garble them):
{terms}

Return ONLY a JSON array, no prose. One object per fix:
{{"at": "<block start time as written in its header>", "from": "<exact substring of that block, long enough to occur once in it>", "to": "<replacement>", "why": "<short reason in Korean>"}}
Return [] if nothing needs fixing.

Transcript:
"""


def split_blocks(text):
    """→ [(stamp, header line, body)], body as written (may span lines)."""
    out = []
    for line in text.splitlines(keepends=True):
        m = HEADER_RE.match(line.strip())
        if m:
            out.append([m.group(1), line, ""])
        elif out:
            out[-1][2] += line
        elif line.strip():
            out.append(["", "", line])
    return out


def ask(prompt, model):
    run = subprocess.run(
        ["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence"],
        input=prompt, capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=900, check=False,
    )
    if run.returncode:
        raise SystemExit(f"claude failed: {run.stderr.strip()[:500]}")
    m = re.search(r"\[.*\]", run.stdout, re.DOTALL)
    return json.loads(m.group()) if m else []


def apply(blocks, edits):
    """Apply edits in place; → rows for the change table."""
    by_stamp = {b[0]: b for b in blocks}
    rows = []
    for e in edits:
        at, old, new = str(e.get("at", "")), str(e.get("from", "")), str(e.get("to", ""))
        block = by_stamp.get(at)
        ok = block is not None and old and old != new and block[2].count(old) == 1
        if ok and block is not None:
            block[2] = block[2].replace(old, new)
        rows.append((at, old, new, str(e.get("why", "")), "applied" if ok else "skipped"))
    return rows


def post_edit(path, course="", model="opus"):
    path = Path(path)
    blocks = split_blocks(path.read_text(encoding="utf-8"))
    terms = sorted({c for c, _ in load_glossary(REPO / "glossary" / "terms.tsv", course or None)})
    rules = RULES.format(course=course or "(unknown)", terms=", ".join(terms))
    rows = []
    for i in range(0, len(blocks), CHUNK):
        part = blocks[i:i + CHUNK]
        rows += apply(part, ask(rules + "".join(h + b for _, h, b in part), model))
    stem = path.name.removesuffix(".md")
    path.with_name(stem + ".post.md").write_text("".join(h + b for _, h, b in blocks), encoding="utf-8")
    table = ["block\tfrom\tto\twhy\tstatus"] + ["\t".join(r) for r in rows]
    path.with_name(stem + ".post.tsv").write_text("\n".join(table) + "\n", encoding="utf-8")
    done = sum(r[4] == "applied" for r in rows)
    print(f"{path.name}: {done} applied, {len(rows) - done} skipped → {stem}.post.md")


def main():
    args = sys.argv[1:]
    model = "opus"
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]
    if not args:
        raise SystemExit(__doc__)
    post_edit(args[0], args[1] if len(args) > 1 else "", model)


if __name__ == "__main__":
    main()
