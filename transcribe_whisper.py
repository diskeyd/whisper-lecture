#!/usr/bin/env python3
"""
Lecture Transcription using Local MLX-Whisper on Apple Silicon

오디오를 통째로 전사한 뒤 반복 환각(hallucination) 필터를 거쳐
1분 단위 타임스탬프 마크다운으로 Obsidian vault에 저장합니다.

전사가 끝나면 원본 오디오를 Google Drive(강의녹음 폴더)와 과목별 NotebookLM
노트북에 올리고, **Drive 업로드가 성공하고 NLM 소스가 ready 가 된 경우에만** 로컬 원본을 삭제합니다.
NotebookLM 노트북이 없는 과목은 전사만 하고 두 업로드를 모두 건너뜁니다.
(기존에 반복 task로 쪼개져 있던 전사·Drive 업로드·NLM 업로드를 한 번의 실행으로 합친 것)

Usage:
  python3 transcribe_whisper.py "~/Downloads/260303 컴퓨터네트워크w1-1.m4a"
  python3 transcribe_whisper.py --dump-json <audio>   # 후처리 전 Whisper 세그먼트 JSON 보존(디버그)
  python3 transcribe_whisper.py --no-upload <audio>   # 전사만 (업로드·삭제 없음)
  python3 transcribe_whisper.py --keep-audio <audio>  # 업로드는 하되 원본은 남김
  python3 transcribe_whisper.py --upload-only <audio> # 전사 건너뛰고 업로드·삭제만 (재시도용)
  python3 transcribe_whisper.py --recheck <audio>     # 저신뢰 구간을 2차 디코딩해 사이드카에 대조(발췌 교정용)
  python3 transcribe_whisper.py --carry-context <audio>  # 창 간 문맥 유지(실험용 — 용어 표기 유지 vs 반복 환각)
  python3 transcribe_whisper.py --carry-context --seed-glossary <audio>  # 용어 사전을 프롬프트로 심는다(실험용)

출력: <stem>.md (전사문, 형식 불변) + <stem>.review.jsonl (세그먼트별 신뢰도·플래그 사이드카).
      사이드카는 한 줄 = 한 세그먼트. flags: low_logprob · maybe_silence · repetitive ·
      term_miss · disagree(2차 디코딩이 1차와 다름 → alt에 2차 결과). 본문은 자동으로 고치지 않는다.

이력: Apple 내장 DictationTranscriber를 "심판"으로 병렬 실행하는 Tier 2가 있었으나
      실효가 거의 없고(+25~36% 소요) 소스(apple_stt.swift)도 유실되어 2026-09-01 제거했습니다.
      측정 근거·설계 원칙·튜닝 값은 아래에 보존되어 있습니다.
        - a private measurement note (not published)
        - a private accuracy guide (not published)
"""

import re
import sys
import json
import time
import shutil
import tempfile
import mimetypes
import subprocess
import unicodedata
from pathlib import Path

try:
    import mlx_whisper
except ImportError:
    print(
        "Error: mlx-whisper is not installed. Run 'pip install mlx-whisper mlx --break-system-packages'"
    )
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────

# Raw lecture transcripts are RAW sources → LLM-Wiki satellite vault (main vault = curated markdown only)
VAULT_ROOT = Path(
    "<WIKI_VAULT>"
)
FILES_ROOT = VAULT_ROOT / "10. Raw Sources"

COURSE_MAP = {
    "네트워크보안": "201.59 네트워크보안",
    "데이터통신": "201.60 데이터통신",
    "시스템보안": "201.62 시스템보안",
    "컴퓨터네트워크": "201.66 컴퓨터네트워크",
    "컴퓨터구조": "201.67 컴퓨터구조",
    "데이터베이스": "201.68 데이터베이스",
    "정보보호": "201.69 정보보호",
    "컴퓨터과학개론": "201.70 컴퓨터과학개론",
    "보안기술최신동향": "201.71 보안기술최신동향",
    "산학프로젝트": "201.72 산학프로젝트1",
    "운영체제": "201.73 운영체제",
    "암호학": "201.74 암호학",
    "알고리즘": "201.75 알고리즘",
    "arvr": "201.76 (기업주도)ARVR메타버스",
    "신기술산업체": "201.77 신기술산업체현장실습2",
    "산업체현장실습": "201.77 신기술산업체현장실습2",  # 별칭
    "오픈소스플랫폼": "201.78 (기업주도)오픈소스 AI플랫폼 엔지니어링",
    "오픈소스프랫폼": "201.78 (기업주도)오픈소스 AI플랫폼 엔지니어링",  # 오타 대비 별칭
    # 과목이 아닌 반복 세미나. NLM 노트북이 없어 업로드는 건너뛰고 원본은 남는다.
    "월간obsidian": "201.19 월간 Obsidian",
}

DEFAULT_LANGUAGE = "ko"

# 원본 오디오 목적지 ─ Drive는 평평한 단일 폴더(기존 수동 업로드와 같은 자리),
# NotebookLM은 과목별 노트북. 노트북 UUID의 정본은 메인 볼트 Subject 노트의
# `- **NotebookLM**:: [NotebookLM](https://notebooklm.google.com/notebook/<uuid>)` 인라인 필드다.
MAIN_VAULT_ROOT = Path(
    "<VAULT>"
)
SUBJECT_DIR = MAIN_VAULT_ROOT / "30. Permanent Notes"
DRIVE_FOLDER_ID = "<DRIVE_FOLDER_ID>"  # 강의녹음 폴더

# NotebookLM 업로드용 MIME. 확장자 추론이 .m4a를 놓치는 경우가 있어 직접 못박는다
# (2026-09-09 실측: audio/mp4로 올린 m4a가 type=media/status=ready로 처리됨).
AUDIO_MIME = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
}

DRIVE_TIMEOUT = 1800  # 40MB대 파일 업로드 여유
NLM_TIMEOUT = 2400  # source add는 업로드만 기다린다 — 서버 처리(ready)는 source wait가 따로 본다
# ready 대기. 실측은 하나뿐이다: 43분·21.8 MB m4a(2026-09-30)가 올린 뒤 몇 분 안에 ready 로 보였다
# (처리 시간 자체는 기록된 적 없음 — 상한만 안다). 상한을 넘기면 원본을 남기고 --upload-only 가
# 같은 소스를 다시 기다리므로, 짧게 잡아도 안전하다.
NLM_READY_TIMEOUT = 600
NLM_READY_INTERVAL = 15
NLM_REQUEST_TIMEOUT = "900"  # CLI 자체의 HTTP 타임아웃(기본 30초는 대용량에 부족)

GROUP_SECONDS = 60  # 출력 타임스탬프를 묶는 단위(초)

# 실행 기록 — 한 파일당 한 줄. data/ 는 gitignore 이고 전사 내용은 들어가지 않는다.
# GPU 점유율(%)은 powermetrics 가 sudo 를 요구해 매 실행마다 잴 수 없다. 대신 단계별
# 소요 시간을 남긴다 — mlx-whisper 가 도는 동안 GPU 는 사실상 포화이므로 전사·재확인
# 초가 곧 GPU 점유 시간이고, 업로드 초는 네트워크라 그와 구분된다.
RUN_LOG = Path(__file__).resolve().parent / "data" / "run-log.jsonl"


def record_run(entry):
    """실행 기록 한 줄을 덧붙인다. 실패해도 전사 결과에는 영향이 없어야 한다."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        entry["cpu_user_s"] = round(usage.ru_utime, 1)
        entry["cpu_sys_s"] = round(usage.ru_stime, 1)
        entry["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠  실행 기록 실패(전사에는 영향 없음): {e}")

# ─── 검토 사이드카 (<stem>.review.jsonl) ──────────────────────────────────────
# 전사문 .md는 사람과 /lecture-note 용이고, 신뢰도·판정은 이 사이드카에 따로 남긴다.
# .md 본문 형식은 절대 건드리지 않는다 (하류 스킬이 깨진다).
# 주의(2026-09-21 실측): avg_logprob·no_speech_prob는 세그먼트가 아니라 **30초 디코딩 창** 단위
# 값이라, 같은 창의 세그먼트는 전부 같은 숫자를 갖는다. 그리고 확신에 찬 오인식
# ("스택 카나리"→"스택 하나리", avg_logprob -0.08)은 이 값으로는 잡히지 않는다.
# 신뢰도 플래그가 잡는 것은 무음 환각·뭉개진 창이고, 용어 오류는 term_miss(용어 사전) 몫이다.
# spelling_fixed 플래그와 `fixed` 필드는 저장 직후 표기 치환이 손댄 구간을 표시한다 —
# 사람 확인 없이 본문을 바꾸는 단계이므로 되짚을 수 있는 기록을 남긴다.
REVIEW_LOGPROB_THRESHOLD = -0.7  # avg_logprob가 이보다 낮으면 low_logprob
REVIEW_NO_SPEECH_THRESHOLD = 0.6  # no_speech_prob가 이보다 높으면 maybe_silence
RECHECK_MIN_DURATION = 1.0  # 이보다 짧은 구간은 재디코딩해도 신호가 없다
RECHECK_MAX_SEGMENTS = 200  # 2차 디코딩 상한 — 플래그가 폭주해도 시간이 터지지 않게
RECHECK_PAD_SECONDS = 0.5  # 앞뒤 여유 — 경계에서 음절이 잘리는 것 방지
RECHECK_TEMPERATURE = 0.2  # 1차(그리디)와 다른 조건으로 디코딩해야 대조가 된다
GLOSSARY_PATH = Path.home() / "VScode/whisper-lecture/glossary/terms.tsv"
GLOSSARY_PROMPT_TERMS = 30  # initial_prompt는 224토큰 제한이라 용어 수를 제한한다
MAX_REPEATED_SEGMENTS = (
    2  # 동일 텍스트 세그먼트의 연속 '추가' 허용 횟수 — 원본 1개 + 2개 = 3개까지 남긴다
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def detect_course(filename):
    """파일명에 포함된 과목 키워드를 찾습니다.

    긴 키가 짧은 키를 포함할 수 있으므로 긴 것부터 검사하고,
    영문 키(arvr 등)의 대소문자는 구분하지 않습니다.
    한글 파일명은 NFC로 맞춘 뒤 비교합니다 — 같은 이름이라도 NFD로 들어오면
    코드에 적힌 NFC 키워드와 한 글자도 일치하지 않아 조용히 미매칭이 됩니다.
    """
    name = unicodedata.normalize("NFC", filename).lower()
    for course in sorted(COURSE_MAP, key=len, reverse=True):
        if course.lower() in name:
            return course
    return None


def choose_language(name=None):
    """전사 언어를 선택합니다.

    name=None   배치 전체에 적용할 언어를 묻습니다. '4. Choose per file'을 고르면 "PER_FILE"을 반환합니다.
    name=<파일명>  PER_FILE 모드에서 파일 하나의 언어를 묻습니다.
    비대화형(파이프·리다이렉트)에서는 묻지 않고 기본값을 씁니다.
    """
    if not sys.stdin.isatty():
        return DEFAULT_LANGUAGE

    if name is None:
        print("\n🌐 Select transcription language:")
        print("  1. Korean (ko) [Default]")
        print("  2. English (en)")
        print("  3. Auto-detect")
        print("  4. Choose per file")
        prompt = "Enter choice (1-4) or press Enter: "
    else:
        print(f"\n🌐 Language for: {name}")
        print("  1. Korean (ko) [Default]   2. English (en)   3. Auto-detect")
        prompt = "Enter choice (1-3) or press Enter: "

    try:
        choice = input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nUsing default: Korean (ko)")
        return "ko"

    if choice == "2":
        return "en"
    if choice == "3":
        return None
    if choice == "4" and name is None:
        return "PER_FILE"
    return "ko"


def check_dependencies():
    """스크립트 실행에 필요한 외부 도구(ffprobe, ffmpeg)가 있는지 확인합니다.

    ffprobe는 get_duration()이 직접 쓰고, ffmpeg는 이 파일에서 직접 부르진 않지만
    mlx-whisper가 오디오를 디코딩할 때 필요하므로 둘 다 검사합니다.
    """
    for tool in ["ffmpeg", "ffprobe"]:
        if not shutil.which(tool):
            print(f"❌ Error: '{tool}'이 시스템에서 발견되지 않았습니다.")
            print("Homebrew를 사용하여 설치해 주세요: brew install ffmpeg")
            sys.exit(1)


def get_duration(path):
    """ffprobe로 오디오 총 재생시간(초)을 구합니다. 실패하면 0.0."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def send_notification(title, message):
    """macOS 시스템 알림을 보냅니다."""
    try:
        # 역슬래시를 먼저 — 나중에 하면 방금 넣은 이스케이프까지 다시 먹는다
        t = title.replace("\\", "\\\\").replace('"', '\\"')
        m = message.replace("\\", "\\\\").replace('"', '\\"')
        cmd = f'display notification "{m}" with title "{t}"'
        subprocess.run(["osascript", "-e", cmd], check=False)
    except Exception:
        pass


# ─── 원본 오디오 처리: Drive · NotebookLM · 삭제 ──────────────────────────────


def run_cli(cmd, timeout):
    """외부 CLI를 실행하고 (성공여부, stdout, 실패사유)를 돌려줍니다."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, "", f"'{cmd[0]}' 명령을 찾을 수 없습니다"
    except subprocess.TimeoutExpired:
        return False, "", f"{timeout}초 안에 끝나지 않았습니다"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, proc.stdout, tail[-1][:200] if tail else f"exit {proc.returncode}"
    return True, proc.stdout, ""


def parse_json_output(out):
    """CLI stdout에서 JSON 객체만 뽑습니다.

    gws는 앞에 'Using keyring backend: …', notebooklm은 진행 로그를 섞어 내보내므로
    '{'마다 파싱을 시도합니다. json.loads와 달리 raw_decode는 JSON 뒤에 로그가 더
    붙어도 성공하고, 로그 줄에 끼어든 짧은 조각 대신 가장 긴 객체를 고릅니다.
    """
    decoder = json.JSONDecoder()
    best, best_len = None, 0
    start = out.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(out, start)
        except json.JSONDecodeError:
            start = out.find("{", start + 1)
            continue
        if end - start > best_len:
            best, best_len = obj, end - start
        start = out.find("{", end)  # 방금 읽은 객체 안쪽은 건너뛴다
    return best


def find_notebook_uuid(course_id):
    """과목 ID('201.66 컴퓨터네트워크')로 Subject 노트를 찾아 NotebookLM UUID를 꺼냅니다.

    한글 파일명은 macOS/iCloud에서 NFD로 저장돼 비교가 어긋나므로 앞의 숫자 코드
    ('201.66')로만 매칭합니다. 같은 과목명이 두 번 개설된 경우(201.17/201.73 운영체제)도
    코드가 갈라주므로 안전합니다.
    """
    code = course_id.split()[0]
    for note in sorted(SUBJECT_DIR.glob(f"{code}*.md")):
        try:
            text = note.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = re.search(
            r"notebooklm\.google\.com/notebook/([0-9a-fA-F-]{36})",
            text,
        )
        if m:
            return m.group(1), note.stem
    return None, None


AUTH_ERROR_HINTS = (
    "invalid_grant",
    "expired or revoked",
    "authentication failed",
    "auth failed",
    "failed to get token",
    "unauthorized",
    "not logged in",
    "please log in",
)
# 맨 401만 인증 실패로 본다 — 소수점이 붙은 초 단위 로그("after 401.0s")를 오탐하지 않는다
HTTP_401 = re.compile(r"(?<![\d.])401(?![\d.])")
# 5xx 는 Google 쪽 일시 오류(2026-09-23 502) — 같은 요청을 다시 보내면 대개 된다
HTTP_5XX = re.compile(r"(?<![\d.])5\d\d(?![\d.])")
# 로그인 화면에서 권한을 일부만 고르면 기존 권한이 그걸로 통째로 바뀐다 → 지금 쓰는 5종을 명시
GWS_SCOPES = ",".join(
    f"https://www.googleapis.com/auth/{s}"
    for s in ("drive", "calendar", "documents", "gmail.readonly", "spreadsheets")
)


def looks_like_auth_error(message):
    """CLI 실패 사유가 '로그인이 풀렸다'인지 판정합니다."""
    low = (message or "").lower()
    return any(hint in low for hint in AUTH_ERROR_HINTS) or bool(HTTP_401.search(low))


def auth_is_valid(tool):
    """지금 이 도구로 API를 부를 수 있는 상태인지 확인합니다."""
    if tool == "gws":
        ok, out, _ = run_cli(["gws", "auth", "status"], 120)
        return bool(ok and (parse_json_output(out) or {}).get("token_valid"))
    ok, out, _ = run_cli(
        ["notebooklm", "auth", "check", "--json", "--test", "--passive"], 180
    )
    return bool(ok and (parse_json_output(out) or {}).get("status") == "ok")


def relogin(tool, message):
    """인증 실패면 그 자리에서 재로그인을 제안합니다. 재시도해도 되면 True.

    전사는 이미 끝난 뒤라 재시도 비용은 업로드뿐입니다. 비대화형이면 묻지 않고
    False를 돌려주고, 원본은 남으므로 나중에 --upload-only로 이어서 하면 됩니다.
    """
    if not looks_like_auth_error(message):
        return False
    login_cmd = (
        ["gws", "auth", "login", "--scopes", GWS_SCOPES]
        if tool == "gws"
        else ["notebooklm", "login"]
    )
    joined = " ".join(login_cmd)
    if not sys.stdin.isatty():
        print(f"   ↳ {tool} 로그인이 풀렸습니다 — `{joined}` 후 --upload-only로 재시도")
        return False

    print(f"   ↳ {tool} 로그인이 풀렸습니다.")
    print(f"      Y = 로그인 후 업로드만 재시도 (다른 터미널에서 이미 `{joined}` 했으면 그대로 Y)")
    try:
        answer = input("      진행할까요? [Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if answer in ("n", "no"):
        return False

    # 다른 터미널에서 이미 로그인했다면 브라우저를 또 띄우지 않고 바로 재시도한다
    if auth_is_valid(tool):
        print("      ✅ 이미 로그인돼 있습니다 — 업로드만 재시도합니다.")
        return True

    # 브라우저가 열리고 사용자 입력을 받으므로 stdio를 그대로 물려준다
    try:
        return subprocess.run(login_cmd).returncode == 0
    except (FileNotFoundError, KeyboardInterrupt):
        return False


def same_filename(a, b):
    """한글 파일명 비교 — macOS/iCloud는 NFD, 서버는 원본 그대로라 정규화 후 비교합니다."""
    return unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)


def drive_existing_id(name):
    """Drive 강의녹음 폴더에 같은 이름의 파일이 이미 있으면 그 id를 돌려줍니다."""
    # Drive의 name 검색은 문자열 완전일치라 NFC/NFD가 다르면 못 찾는다 → 두 형태 모두 질의
    variants = []
    for form in ("NFC", "NFD"):
        v = unicodedata.normalize(form, name)
        if v not in variants:
            variants.append(v)
    name_clause = " or ".join(
        "name = '{}'".format(v.replace("\\", "\\\\").replace("'", "\\'"))
        for v in variants
    )
    params = json.dumps(
        {
            "q": (
                f"({name_clause}) and '{DRIVE_FOLDER_ID}' in parents "
                "and trashed = false"
            ),
            "fields": "files(id,name)",
        },
        ensure_ascii=False,
    )
    ok, out, _ = run_cli(
        ["gws", "drive", "files", "list", "--params", params], DRIVE_TIMEOUT
    )
    if not ok:
        return None  # 조회 실패는 '없음'으로 보고 업로드를 진행한다
    data = parse_json_output(out) or {}
    files = data.get("files") or []
    return files[0].get("id") if files else None


def upload_to_drive(path, retry=True):
    """Drive 강의녹음 폴더에 올립니다. (file_id, 메모) / 실패 시 (None, 사유)."""
    existing = drive_existing_id(path.name)
    if existing:
        return existing, "이미 있음"

    ok, out, err = run_cli(
        [
            "gws",
            "drive",
            "+upload",
            str(path),
            "--parent",
            DRIVE_FOLDER_ID,
            "--format",
            "json",
        ],
        DRIVE_TIMEOUT,
    )
    data = parse_json_output(out) or {}
    code = ""
    if isinstance(data.get("error"), dict):
        # gws 는 message 가 빈 오류 JSON 을 낼 때가 있다. 예전에는 그 빈 값이 stderr 의
        # `error[...]` 줄을 덮어써서 사유가 공란으로 찍혔다(2026-09-22·23 두 번).
        e = data["error"]
        code = str(e.get("code") or "")
        stderr_line = err if err.startswith("error") else ""  # `error[api]: …` 줄
        err = str(e.get("message") or e.get("reason") or stderr_line or e.get("code") or err)[:200]
        ok = False
    if not ok:
        # 일시 오류면 한 번만 다시 올린다. 실제로는 올라갔는데 5xx 가 온 경우는
        # 재시도 첫머리의 drive_existing_id 가 잡아서 두 번 올리지 않는다
        if retry and HTTP_5XX.search(f"{code} {err}"):
            print(f"   ↳ Drive 일시 오류({code or err}) — 10초 뒤 한 번 더 시도")
            time.sleep(10)
            return upload_to_drive(path, retry=False)
        # 토큰 만료라면 로그인만 다시 하고 업로드부터 이어간다 (전사는 재실행하지 않음)
        if retry and relogin("gws", err):
            return upload_to_drive(path, retry=False)
        return None, err or "gws 가 사유 없이 실패"
    file_id = data.get("id") or (data.get("file") or {}).get("id")
    return (file_id, "업로드") if file_id else (None, "응답에 파일 id가 없습니다")


def upload_to_notebooklm(path, uuid, allow_relogin=True):
    """과목 노트북에 오디오 소스를 추가합니다. (source_id, 메모) / 실패 시 (None, 사유).

    업로드된 파일의 소스 제목은 서버가 파일명으로 잡으므로, 같은 파일명이
    이미 정상 소스로 들어가 있으면 재실행에서 건너뜁니다(소스 300개 상한 보호).
    """
    ok, out, _ = run_cli(["notebooklm", "source", "list", "-n", uuid, "--json"], 300)
    if ok:
        data = parse_json_output(out) or {}
        for src in data.get("sources") or []:
            title = src.get("title") or ""
            if not same_filename(title, path.name):
                continue
            status = src.get("status")
            if status == "ready":
                return src.get("id"), "이미 있음"
            if status == "error":
                break  # 실패한 소스는 남아 있어도 없는 셈 치고 다시 올린다
            # processing 등 미완료 — 다시 올리지 않고 ready 를 기다린다
            return wait_until_ready(src.get("id"), uuid, "이미 있음")

    mime = (
        AUDIO_MIME.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "audio/mp4"
    )
    ok, out, err = run_cli(
        [
            "notebooklm",
            "source",
            "add",
            str(path),
            "--type",
            "file",
            "--mime-type",
            mime,
            "-n",
            uuid,
            "--request-timeout",
            NLM_REQUEST_TIMEOUT,
            "--json",
        ],
        NLM_TIMEOUT,
    )
    data = parse_json_output(out) or {}
    # notebooklm은 처리 실패도 exit 0 + {"error": true, ...}로 돌려준다
    if data.get("error"):
        err = str(data.get("message", "처리 실패"))[:200]
        ok = False
    if not ok:
        if allow_relogin and relogin("notebooklm", err):
            return upload_to_notebooklm(path, uuid, allow_relogin=False)
        return None, err
    source_id = (data.get("source") or {}).get("id")
    if not source_id:
        return None, "응답에 source id가 없습니다"
    return wait_until_ready(source_id, uuid, "업로드")


def wait_until_ready(source_id, uuid, note):
    """소스가 ready 가 될 때까지 기다립니다. (source_id, 메모) / 아니면 (None, 사유).

    source add 는 업로드가 끝나면 돌아오고 서버 처리는 그 뒤에 이어지므로, id 만으로는
    원본을 지울 근거가 되지 않는다. ready 가 아니면 원본을 남긴다.
    """
    # ponytail: 배치는 파일마다 여기서 최대 NLM_READY_TIMEOUT 만큼 멈춘다. 배치가 길어지면
    # 대기 소스를 모아 배치 끝에서 한 번에 기다리도록 바꾼다.
    _, out, err = run_cli(
        [
            "notebooklm",
            "source",
            "wait",
            source_id,
            "-n",
            uuid,
            "--timeout",
            str(NLM_READY_TIMEOUT),
            "--interval",
            str(NLM_READY_INTERVAL),
            "--json",
        ],
        NLM_READY_TIMEOUT + 120,  # CLI 가 먼저 끝나야 timeout 결과(JSON)를 읽는다
    )
    status = (parse_json_output(out) or {}).get("status")
    if status == "ready":
        return source_id, note
    if status == "timeout":
        return None, f"NLM 처리 중({NLM_READY_TIMEOUT}초 초과) — 잠시 후 --upload-only로 재시도"
    return None, f"NLM 처리 실패({status or err or '응답 없음'})"


def dispatch_audio(input_path, course_id, keep_audio=False):
    """Drive → NotebookLM 순으로 올리고, 둘 다 성공하면 원본을 삭제합니다. (성공여부, 요약)

    한 곳이라도 실패하면 원본을 남깁니다 — 로컬 파일이 유일한 재시도 경로이기 때문입니다.
    과목을 못 찾았거나 노트북이 없으면 두 업로드를 모두 건너뛰고 원본도 그대로 둡니다.
    """
    print("\n📤 원본 오디오 처리")

    # 노트북부터 확인한다 — 노트북이 없는 과목은 전사만 하고 Drive·NLM 둘 다 건너뛴다
    # (사용자 결정 2026-09-09: 반쪽 업로드로 원본이 흩어지는 것보다 로컬에 두는 게 낫다)
    uuid, subject_note = find_notebook_uuid(course_id) if course_id else (None, None)
    if not uuid:
        reason = "과목 미매칭" if not course_id else f"{course_id} 노트북 없음"
        print(f"   • Drive      : ⏭  건너뜀 ({reason})")
        print(f"   • NotebookLM : ⏭  건너뜀 ({reason})")
        print("   • 원본       : 📌 유지 (전사만 수행)")
        return True, f"업로드 생략({reason}) · 전사만 · 원본 유지"

    drive_id, drive_note = upload_to_drive(input_path)
    if drive_id:
        print(f"   • Drive      : ✅ {drive_note} ({drive_id})")
    else:
        print(f"   • Drive      : ❌ {drive_note}")

    source_id, nlm_note = upload_to_notebooklm(input_path, uuid)
    if source_id:
        print(f"   • NotebookLM : ✅ {nlm_note} → {subject_note}")
    else:
        print(f"   • NotebookLM : ❌ {nlm_note}")

    if drive_id and source_id and not keep_audio:
        if input_path.is_symlink():
            print("   • 원본       : ⏭  심볼릭 링크라 삭제하지 않음")
            return True, "Drive ✅ · NLM ✅ · 원본 유지(심링크)"
        try:
            input_path.unlink()
            print(f"   • 원본       : 🗑  삭제 ({input_path.name})")
            return True, "Drive ✅ · NLM ✅ · 원본 삭제"
        except OSError as e:
            print(f"   • 원본       : ⚠️  삭제 실패 — {e}")
            return True, "Drive ✅ · NLM ✅ · 원본 삭제 실패"

    if drive_id and source_id:
        print("   • 원본       : 📌 유지 (--keep-audio)")
        return True, "Drive ✅ · NLM ✅ · 원본 유지"

    print(
        f"   • 원본       : 📌 유지 — 재시도: {sys.argv[0]} --upload-only '{input_path}'"
    )
    return False, (
        f"Drive {'✅' if drive_id else '❌'} · "
        f"NLM {'✅' if source_id else '❌'} · 원본 유지"
    )


def run_dispatch(input_path, course_id, keep_audio=False):
    """dispatch_audio를 감싸 예외까지 '업로드 실패'로 돌려줍니다.

    업로드 단계에서 터진 예외가 배치 전체를 멈추거나 '전사 실패'로 둔갑하지 않게 합니다.
    """
    try:
        return dispatch_audio(input_path, course_id, keep_audio=keep_audio)
    except Exception as e:
        print(f"\n❌ 업로드 단계 에러: {e}")
        return False, f"업로드 에러({str(e)[:50]}) · 원본 유지"


# ─── Tier 1: 반복 환각 필터 ───────────────────────────────────────────────────


def remove_consecutive_phrase_repeats(
    text, min_repeats=3, max_phrase_len=5, min_single_repeats=4
):
    """동일한 구절(2~5단어)이 3회 이상, 또는 단일 단어가 4회 이상 연속 반복되는 경우 하나만 남깁니다.
    단일 단어 임계값을 높게 둬서 자연스러운 반복('네 네 네')은 보존하고, 무음 구간 환각(수십 회 반복)만 제거합니다.
    잔여 환각은 다운스트림 AI 노트 정리 단계에서 재검증되므로 보존 우선 정책을 따릅니다."""
    words = text.split()
    if not words:
        return text

    n = len(words)
    i = 0
    cleaned_words = []

    while i < n:
        matched = False
        # 짧은 주기부터 확인해야 '가 나'×10이 '가 나 가 나'(2주기)를 1회분으로 보고
        # 절반만 지우는 일이 없다 (_long_repeat_flags와 같은 이유)
        for length in range(1, max_phrase_len + 1):
            if i + length > n:
                break

            phrase = words[i : i + length]
            repeats = 1
            while True:
                next_start = i + repeats * length
                next_end = next_start + length
                if next_end <= n and words[next_start:next_end] == phrase:
                    repeats += 1
                else:
                    break

            threshold = min_single_repeats if length == 1 else min_repeats
            if repeats >= threshold:
                cleaned_words.extend(phrase)
                i += repeats * length
                matched = True
                break

        if not matched:
            cleaned_words.append(words[i])
            i += 1

    return " ".join(cleaned_words)


# 긴 구절 반복 ─ Whisper large-v3의 대표 환각은 9~14단어 문장이 2~3회 그대로 반복되는 형태로,
# 위 필터(≤5단어·3회)에는 걸리지 않는다. 긴 구절은 자연 반복 확률이 매우 낮아 2회부터 제거한다.


def _long_repeat_flags(words, min_words=6, max_words=20, min_repeats=2):
    """연속 반복되는 긴 구절의 2회째 이후 단어에 삭제 플래그를 세웁니다. (flags, 제거된 반복 수)"""
    n = len(words)
    drop = [False] * n
    removed = 0
    i = 0
    while i < n:
        matched = False
        # 짧은 주기부터 확인해야 'P P P P'가 'PP'×2로 잡혀 절반만 지워지는 일이 없다
        for length in range(min_words, max_words + 1):
            if i + 2 * length > n:
                continue
            phrase = words[i : i + length]
            repeats = 1
            while True:
                s = i + repeats * length
                if s + length <= n and words[s : s + length] == phrase:
                    repeats += 1
                else:
                    break
            if repeats >= min_repeats:
                for k in range(i + length, i + repeats * length):
                    drop[k] = True
                removed += repeats - 1
                i += repeats * length
                matched = True
                break
        if not matched:
            i += 1
    return drop, removed


def remove_long_phrase_repeats(text, min_words=6, max_words=20, min_repeats=2):
    """텍스트 한 덩어리에서 긴 구절 반복을 제거합니다(더 이상 지울 것이 없을 때까지 반복). (text, 제거 수)"""
    words = text.split()
    total = 0
    while words:
        drop, removed = _long_repeat_flags(words, min_words, max_words, min_repeats)
        if not removed:
            break
        words = [w for w, d in zip(words, drop) if not d]
        total += removed
    return (" ".join(words), total) if total else (text, 0)


def remove_long_phrase_repeats_in_segments(segments):
    """세그먼트 경계를 넘어 반복되는 긴 구절을 전체 텍스트 수준에서 제거합니다. (segments, 제거 수)"""
    words, owner = [], []
    for si, seg in enumerate(segments):
        for w in seg.get("text", "").split():
            words.append(w)
            owner.append(si)
    removed = 0
    while words:
        drop, r = _long_repeat_flags(words)
        if not r:
            break
        words = [w for w, d in zip(words, drop) if not d]
        owner = [o for o, d in zip(owner, drop) if not d]
        removed += r
    if not removed:
        return segments, 0
    kept = {}
    for w, o in zip(words, owner):
        kept.setdefault(o, []).append(w)
    out = []
    for si, seg in enumerate(segments):
        if si in kept:
            seg["text"] = " ".join(kept[si])
            out.append(seg)
    return out, removed


# ─── 출력 포맷 ────────────────────────────────────────────────────────────────


def clean_and_group_segments(segments):
    """
    1. Whisper 환각 패턴(끝부분 무음 구간에서 '됐습니다' 등 연속 중복) 필터링
    2. 1분 단위로 병합하여 타임스탬프 부여
    """
    cleaned = []
    consecutive_duplicates = 0
    last_text = None

    # 1. 환각 필터링
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # 단어/구절 단위 반복 환각 제거
        text = remove_consecutive_phrase_repeats(text)
        if not text:
            continue

        # 동일 텍스트 연속 반복 감지
        if text == last_text:
            consecutive_duplicates += 1
            if consecutive_duplicates > MAX_REPEATED_SEGMENTS:
                # 4번째 사본부터 스킵 (주로 무음 구간 환각)
                continue
        else:
            consecutive_duplicates = 0

        last_text = text
        seg["text"] = text
        cleaned.append(seg)

    # 2. 1분 단위 그룹화
    grouped = {}
    for seg in cleaned:
        start_time = seg.get("start", 0.0)
        minute_idx = int(start_time // GROUP_SECONDS)
        grouped.setdefault(minute_idx, []).append(seg.get("text", "").strip())

    # 3. 마크다운 포맷팅
    def fmt(secs, with_hours):
        secs = int(secs)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        if with_hours:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    formatted_lines = []
    if not grouped:
        return ""

    # 실제 마지막 세그먼트 종료시각 → 마지막 구간 라벨 오버슛 방지
    total_end = int(max((seg.get("end", 0.0) for seg in cleaned), default=0.0))

    max_minute = max(grouped.keys())
    # 한 시간을 넘는 녹음은 전체를 hh:mm:ss로 — '59:00 - 01:00:00'처럼 섞이지 않게
    with_hours = (max_minute + 1) * GROUP_SECONDS >= 3600
    for i in range(max_minute + 1):
        texts = grouped.get(i, [])
        if not texts:
            continue

        start_secs = i * GROUP_SECONDS
        end_secs = (i + 1) * GROUP_SECONDS
        if total_end:
            end_secs = min(end_secs, max(total_end, start_secs + 1))

        ts = f"**{fmt(start_secs, with_hours)} - {fmt(end_secs, with_hours)}**"
        combined_text = " ".join(texts)
        combined_text = remove_consecutive_phrase_repeats(combined_text)
        combined_text, _ = remove_long_phrase_repeats(combined_text)
        formatted_lines.append(f"{ts}\n{combined_text}")

    return "\n\n".join(formatted_lines)


# ─── 검토 사이드카 ────────────────────────────────────────────────────────────


def load_glossary(course_id=None):
    """용어 사전을 읽습니다. → [(correct, [wrong, ...]), ...]

    형식(TSV, '#'로 시작하는 줄은 주석):
        정답용어<TAB>오답1|오답2<TAB>과목ID(선택, 비면 전 과목 공통)

    파일이 없으면 빈 목록 — 용어 사전은 선택 사항이고, 없다고 전사가 막히면 안 된다.
    """
    if not GLOSSARY_PATH.exists():
        return []
    entries = []
    try:
        for line in GLOSSARY_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            correct = unicodedata.normalize("NFC", parts[0].strip())
            if not correct:
                continue
            scope = parts[2].strip() if len(parts) > 2 else ""
            if scope and course_id and scope != course_id:
                continue
            wrong = [
                unicodedata.normalize("NFC", w.strip())
                for w in (parts[1] if len(parts) > 1 else "").split("|")
                if w.strip()
            ]
            entries.append((correct, wrong))
    except OSError as e:
        print(f"⚠  용어 사전 읽기 실패({GLOSSARY_PATH}): {e}")
        return []
    return entries


def glossary_prompt(entries):
    """용어 사전을 Whisper initial_prompt 문장으로 만듭니다.

    프롬프트의 문체가 출력 문체에 전염되므로, 나열이 아니라 구두점 있는
    한국어 문장 형태로 씁니다. 224토큰 제한 때문에 앞쪽 N개만 씁니다.
    """
    terms = [c for c, _ in entries][:GLOSSARY_PROMPT_TERMS]
    if not terms:
        return None
    return "이 강의에서는 " + ", ".join(terms) + " 같은 용어를 다룹니다."


# 표기 치환 — 같은 뜻인데 노트에 쓸 수 없게 적힌 것을 고친다.
# 2026-09-22 실측: 이 파이프라인은 5분 발췌에서 "ACK"를 12번 전부 한글("에크·액트·액")로
# 적었다. `condition_on_previous_text=True` 로 되돌려도, 75분 전체에 용어 사전을
# initial_prompt 로 심어도 고쳐지지 않았다(71/71 한글). 인식 단계로는 해결되지 않으므로
# 저장 직후 텍스트에서 고친다 — 정답본 대조로 CER 0.127→0.104, 용어 오인식률 0.824→0.176.
# 상용 API 들이 이 문제에 보장하는 수단도 같은 방식(AWS DisplayAs, Deepgram replace)이다.
#
# 짧은 표기의 무맥락 치환은 진짜 한국어를 부순다("참치액젓"·"혈액을"). 그래서 한글 2음절
# 이하는 조사나 동반어가 붙어야, 3음절 이상은 다른 낱말에 붙지만 않으면 치환하고,
# 라틴 표기는 뒤에 영숫자가 오면 건너뛴다. 규칙과 회귀 테스트는
# scripts/fix_terms.py 와 tests/ 에 있다.
ALIASES_PATH = Path.home() / "VScode/whisper-lecture/glossary/aliases.tsv"


def _replace_guarded():
    """scripts/textnorm.py 의 공용 치환기를 늦게 가져온다.

    예전에는 이 파일이 같은 규칙을 따로 구현해 두었는데, 2026-09-22 에 한쪽만
    어미 목록을 넓히는 바람에 파이프라인이 `인터럴트입니다` 같은 자리를 조용히
    건너뛰었다. 이제 구현은 scripts/textnorm.py 한 곳뿐이고 테스트도 거기 붙어 있다.

    최상단이 아니라 호출 시점에 임포트한다 — scripts/ 가 없거나 움직여도 이미 끝난
    전사가 실패로 바뀌면 안 된다. 호출부가 전부 try 안에 있는 이유도 같다.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from textnorm import replace_guarded

    return replace_guarded


def load_spelling_map(course_id=None):
    """[(잘못 적힌 표기, 바른 표기), ...] — 긴 것부터. 용어 사전 + 표기 이명 표."""
    pairs = {}
    for correct, wrong in load_glossary(course_id):
        for w in wrong:
            pairs.setdefault(w, correct)
    if ALIASES_PATH.exists():
        try:
            for line in ALIASES_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                canonical = unicodedata.normalize("NFC", parts[0].strip())
                for v in (parts[1] if len(parts) > 1 else "").split("|"):
                    v = unicodedata.normalize("NFC", v.strip())
                    if v and canonical:
                        pairs.setdefault(v, canonical)
        except OSError as e:
            print(f"⚠  표기 이명 표 읽기 실패: {e}")
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def _normalize_for_compare(text):
    """비교 전용 정규화 — NFC로 맞추고 공백·문장부호를 버립니다.

    macOS/iCloud가 한글을 NFD로 저장하기 때문에 정규화 없이 비교하면
    같은 글자가 자모 단위로 어긋나 전부 불일치로 잡힙니다.
    """
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[\s.,!?·…\"'()\[\]{}~\-—:;]", "", text)


def build_review_records(raw_snapshot, kept_segments, entries):
    """원본 세그먼트 + 반복 필터 결과 + 용어 사전 → 검토 레코드 목록."""
    kept_text = {id(s): s.get("text", "").strip() for s in kept_segments}
    records = []
    dup_run, last_raw = 0, None
    for seg, raw_text in raw_snapshot:
        text = kept_text.get(id(seg), "")
        flags = []

        # clean_and_group_segments가 버리는 연속 동일 세그먼트(4번째 사본부터)는
        # kept_segments에 흔적이 남지 않으므로 같은 규칙을 여기서 다시 센다.
        dup_run = dup_run + 1 if raw_text and raw_text == last_raw else 0
        last_raw = raw_text
        if dup_run > MAX_REPEATED_SEGMENTS:
            text = ""

        avg_logprob = seg.get("avg_logprob")
        no_speech = seg.get("no_speech_prob")
        # NaN 은 디코딩이 무너진 창이다(2026-09-23 알고리즘 34:47, 33초가 "그렇죠" 3개와
        # 빈 세그먼트 70개로 남음). NaN 은 어떤 비교도 거짓이라 예전엔 가장 나쁜 구간이
        # 플래그에서 빠졌다. 빈 텍스트까지 표시하면 교정 시트가 수십 줄로 넘치므로 글자가
        # 남은 세그먼트만 — 그것만으로 그 구간이 시트 맨 앞에 올라온다.
        collapsed = avg_logprob != avg_logprob and bool(text or raw_text)
        if collapsed or (avg_logprob is not None and avg_logprob < REVIEW_LOGPROB_THRESHOLD):
            flags.append("low_logprob")
        if no_speech is not None and no_speech > REVIEW_NO_SPEECH_THRESHOLD:
            flags.append("maybe_silence")
        if text != raw_text:
            # 통째로 삭제됐거나(text == "") 단어가 깎인 구간 — 이음새 결함이 생기는 자리
            flags.append("repetitive")

        haystack = _normalize_for_compare(text or raw_text)
        term = None
        for correct, wrong in entries:
            if any(_normalize_for_compare(w) in haystack for w in wrong):
                flags.append("term_miss")
                term = correct  # 어떤 용어가 깨졌는지 — 검토자가 바로 고칠 수 있게
                break

        records.append(
            {
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": text,
                "raw_text": raw_text if text != raw_text else None,
                "avg_logprob": avg_logprob,
                "no_speech_prob": no_speech,
                "compression_ratio": seg.get("compression_ratio"),
                "flags": flags,
                "term": term,
                "alt": None,
            }
        )
    return records


def recheck_flagged(input_path, records, entries, language):
    """플래그가 걸린 구간만 다른 조건으로 다시 디코딩해 대조합니다.

    1차와 다르면 `alt`에 담고 `disagree`를 붙일 뿐, **본문을 자동으로 바꾸지 않습니다**.
    엔진이 둘일 때 임의로 한쪽을 택하면 오히려 오류율이 올라간다는 것이
    2026-08-26 측정에서 확인된 사항입니다(가이드 문서 '왜 2-엔진 투표가 안 되는가').

    발췌 구간은 5분보다 짧아 단일 창이므로, 여기서는 initial_prompt가 실제로 먹습니다
    (전체 전사에서 initial_prompt가 무효인 것은 창이 여러 개이기 때문).
    """
    targets = [
        r
        for r in records
        if r["flags"] and r["text"] and (r["end"] - r["start"]) >= RECHECK_MIN_DURATION
    ]
    if not targets:
        return 0
    if len(targets) > RECHECK_MAX_SEGMENTS:
        print(
            f"⚠  재확인 대상 {len(targets)}건 → 신뢰도 낮은 순 {RECHECK_MAX_SEGMENTS}건만 처리합니다."
        )
        targets.sort(key=lambda r: (r["avg_logprob"] if r["avg_logprob"] is not None else 0.0))
        targets = targets[:RECHECK_MAX_SEGMENTS]

    prompt = glossary_prompt(entries)
    disagreements = 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_recheck_"))
    try:
        for i, rec in enumerate(targets, 1):
            clip = tmp_dir / f"seg_{i:04d}.wav"
            start = max(0.0, rec["start"] - RECHECK_PAD_SECONDS)
            length = (rec["end"] - rec["start"]) + RECHECK_PAD_SECONDS * 2
            try:
                subprocess.run(
                    # 재인코딩(-c copy 아님) — 구간을 초 단위로 정확히 잘라야 대조가 성립한다
                    ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.2f}",
                     "-t", f"{length:.2f}", "-i", str(input_path),
                     "-ac", "1", "-ar", "16000", "-y", str(clip)],
                    check=True,
                    capture_output=True,
                )
                second = mlx_whisper.transcribe(
                    str(clip),
                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                    verbose=False,
                    language=language,
                    condition_on_previous_text=False,
                    temperature=RECHECK_TEMPERATURE,
                    initial_prompt=prompt,
                )
            except Exception as e:
                rec["flags"].append("recheck_failed")
                print(f"⚠  재확인 실패({rec['start']:.1f}s): {str(e)[:60]}")
                continue
            finally:
                clip.unlink(missing_ok=True)

            alt = str(second.get("text") or "").strip()
            # 2차가 빈 문자열인 것도 불일치다 — 1차가 무음에서 지어낸 환각일 때 흔한 모양
            # ("구독과 좋아요를 눌러주세요" 류). 그래서 alt가 비었는지는 조건에서 보지 않는다.
            if _normalize_for_compare(alt) != _normalize_for_compare(rec["text"]):
                rec["alt"] = alt
                rec["flags"].append("disagree")
                disagreements += 1
            print(f"\r🔁 재확인 {i}/{len(targets)} (불일치 {disagreements})", end="")
    finally:
        print()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return disagreements


def write_review_jsonl(output_path, records):
    """검토 레코드를 전사문 옆 <stem>.review.jsonl로 저장합니다.

    JSONL인 이유: 한 줄 = 한 세그먼트라 grep·스트리밍 파싱이 되고,
    중간이 잘려도 나머지 줄이 살아 있으며, LLM에 그대로 먹이기 좋습니다.
    """
    path = output_path.with_suffix(".review.jsonl")
    try:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"⚠  검토 파일 저장 실패: {e}")
        return None
    return path


# ─── 전사 파이프라인 ──────────────────────────────────────────────────────────


def resolve_output_path(input_path, create_dir=True):
    """파일명에서 과목을 감지해 저장 경로를 정합니다. (출력 경로, 과목 ID)

    과목이 잡히면 vault의 해당 과목 폴더에, 못 잡으면 입력 파일과 같은 폴더에 저장합니다.
    create_dir=False는 경로만 계산합니다 — 전사하지 않는 --upload-only에서 빈 과목
    폴더를 vault에 만들지 않기 위해서입니다.
    """
    course = detect_course(input_path.name)
    if course:
        course_id = COURSE_MAP[course]
        output_dir = FILES_ROOT / course_id
        if create_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📚 Course   : {course_id}")
    else:
        course_id = None
        output_dir = input_path.parent
        print("\n📚 Course   : 매칭 없음 (입력 파일과 같은 폴더에 저장)")
    return output_dir / (input_path.stem + ".md"), course_id


def transcribe_file(
    input_path,
    language=None,
    dump_json=False,
    upload=True,
    keep_audio=False,
    recheck=False,
    carry_context=False,
    seed_glossary=False,
):
    output_path, course_id = resolve_output_path(input_path)
    print(f"📁 Output   : {output_path}")
    print(f"🌐 Language : {language if language else 'Auto-detect'}")

    duration = get_duration(input_path)
    print(f"⏱  Duration  : {duration / 60:.1f} min")

    print("🤖 Transcribing locally using MLX-Whisper (large-v3)...")
    start_time = time.time()

    try:
        # 1시간이 넘는 파일도 청크로 쪼갤 필요 없이 통째로 전사합니다.
        # condition_on_previous_text=False: 이전 창의 출력을 프롬프트로 넘기지 않아 반복 환각 루프를 차단
        # (5분 샘플에서 환각 3건→0건, 실제 내용 손실 없음, 10% 빠름).
        # beam_size는 mlx-whisper가 미지원(greedy 전용)이므로 절대 추가하지 말 것.
        # --carry-context(=condition_on_previous_text=True)는 측정용 실험 스위치다.
        # 2026-09-22 실측: 문맥을 끊으면 창마다 표기 판단이 초기화되어 약어가 깨진다
        # (같은 5분에서 "ACK"가 12/12 한글로 나왔고, 문맥을 유지한 26-1 세대는 15/15 정확).
        # 반면 문맥을 유지하면 반복 환각 루프가 생기는데, 그 위험은 이제 긴 구절 반복
        # 필터가 막아 준다 — 두 조건을 같이 켠 조합이 아직 측정되지 않아 이 플래그로 잰다.
        # --seed-glossary: 용어 사전을 initial_prompt 로 넣어 표기의 첫 기준을 심는다.
        # mlx-whisper 는 initial_prompt 를 all_tokens 앞에 한 번 넣고 창마다
        # all_tokens[prompt_reset_since:] 를 프롬프트로 쓴다. condition_on_previous_text=False 면
        # prompt_reset_since 가 매 창 끝으로 밀려 프롬프트가 첫 창에서만 살아 있고,
        # True 면 끝까지 따라간다 — 그래서 이 두 플래그는 같이 써야 의미가 있다.
        seed = glossary_prompt(load_glossary(course_id)) if seed_glossary else None
        if seed:
            print(f"🌱 용어 프롬프트: {seed[:60]}…")
        result = mlx_whisper.transcribe(
            str(input_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            verbose=False,
            language=language,
            condition_on_previous_text=carry_context,
            initial_prompt=seed,
        )

        segments = result.get("segments", [])
        # 검토 사이드카용 원문 스냅샷 — 아래 반복 필터들이 세그먼트 dict를 제자리에서
        # 고치므로, 필터를 타기 전에 떠 두지 않으면 '무엇이 깎였는지'를 알 수 없다.
        raw_snapshot = [(seg, (seg.get("text") or "").strip()) for seg in segments]

        if (
            dump_json
        ):  # 디버그: 후처리 이전의 Whisper 원본 세그먼트 보존 (오프라인 재처리용)
            try:
                output_path.with_suffix(".whisper.json").write_text(
                    json.dumps({"segments": segments}, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
            except OSError as e:
                print(f"⚠  Whisper JSON 저장 실패: {e}")

        # Tier 1: 세그먼트 경계를 넘는 긴 구절 반복을 단어 단위로 먼저 깔끔히 제거
        segments, n_long = remove_long_phrase_repeats_in_segments(segments)
        if n_long:
            print(f"🧹 긴 구절 반복 제거: {n_long}건")

        # 1분 단위 타임스탬프 결합 및 정리
        formatted_text = clean_and_group_segments(segments)

        if not formatted_text.strip():
            print("⚠  Warning: 전사된 텍스트가 비어 있어 파일을 저장하지 않았습니다.")
            send_notification("전사 경고", "텍스트가 비어 전사를 완료하지 못했습니다.")
            return False, "전사 실패 · 원본 유지"

        # 저장 직후 표기 치환 — 실패해도 전사는 이미 끝난 것으로 취급한다.
        try:
            fixed, applied, _declined = _replace_guarded()(
                formatted_text, load_spelling_map(course_id)
            )
            if applied:
                formatted_text = fixed
                print(f"🔤 표기 치환: {len(applied)}건")
        except Exception as e:
            print(f"⚠  표기 치환 생략({e}) — 전사문은 그대로 저장됩니다")

        output_path.write_text(formatted_text, encoding="utf-8")

        elapsed = time.time() - start_time
        speed_str = (
            f" ({duration / elapsed:.1f}x speed)"
            if duration > 0 and elapsed > 0
            else ""
        )
        print(f"\n✅ Completed in {elapsed:.1f}s{speed_str}")
        print(f"💾 Saved: {output_path}")

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        send_notification(
            "전사 에러 발생", f"{input_path.name} 전사 도중 에러: {str(e)[:50]}"
        )
        return False, "전사 실패 · 원본 유지"

    # 검토 사이드카 — 전사 try 밖, 업로드 앞. 밖에 두는 이유: 여기서 난 예외가 이미 저장된
    # 전사를 '실패'로 뒤집으면 안 된다. 앞에 두는 이유: 2차 디코딩이 원본 오디오를 읽는데
    # 업로드 단계가 성공하면 원본이 삭제된다.
    recheck_seconds, n_rechecked = 0.0, 0
    try:
        entries = load_glossary(course_id)
        records = build_review_records(raw_snapshot, segments, entries)
        recheck_started = time.time()
        n_disagree = (
            recheck_flagged(input_path, records, entries, language) if recheck else 0
        )
        recheck_seconds = time.time() - recheck_started
        n_rechecked = sum(1 for r in records if r.get("alt") is not None)
        # 전사문에는 치환이 적용됐으므로 사이드카의 text 도 같이 맞춘다 — 안 맞추면 교정 시트가
        # 전사문과 다른 표기를 보여준다. 그리고 자동으로 바꾼 자리는 `fixed` 에 남긴다:
        # 치환은 사람 확인 없이 본문을 고치므로, 어디를 고쳤는지 되짚을 수 있어야 한다.
        pairs = load_spelling_map(course_id)
        replace_guarded = _replace_guarded()
        for rec in records:
            for field in ("text", "raw_text", "alt"):
                value = rec.get(field)
                if not value:
                    continue
                fixed_value, applied, _declined = replace_guarded(value, pairs)
                if applied:
                    rec[field] = fixed_value
                    if field == "text":
                        rec.setdefault("flags", []).append("spelling_fixed")
                        # 치환기가 실제로 바꾼 것만. 예전에는 문자열을 대조해 추론했는데,
                        # 긴 변형에 먹혀 한 번도 안 쓰인 짧은 변형까지 적히고(과다), 같은
                        # 낱말의 다른 출현이 보류되면 통째로 빈 목록이 됐다(과소).
                        rec["fixed"] = [[v, c] for v, c, _at in applied]
        review_path = write_review_jsonl(output_path, records)
        if review_path:
            n_flagged = sum(1 for r in records if r["flags"])
            print(
                f"🔎 Review : {review_path.name} — 세그먼트 {len(records)}"
                f" · 플래그 {n_flagged} · 불일치 {n_disagree}"
                + ("" if recheck else " (재확인 생략)")
            )
    except Exception as e:
        print(f"⚠  검토 파일 단계 실패(전사문은 정상 저장됨): {e}")

    # 업로드는 try 밖에서 — 여기서 난 예외를 '전사 실패'로 보고하면 이미 저장된
    # 전사문이 없는 것처럼 보인다. 전사가 성공했을 때만 여기까지 온다.
    # 업로드가 성공하면 원본이 삭제되므로 크기는 그 전에 읽는다.
    try:
        source_megabytes = round(input_path.stat().st_size / 1e6, 1)
    except OSError:
        source_megabytes = None

    upload_started = time.time()
    if upload:
        ok, audio_note = run_dispatch(input_path, course_id, keep_audio=keep_audio)
    else:
        ok, audio_note = True, "업로드 생략(--no-upload) · 원본 유지"
    upload_seconds = time.time() - upload_started

    record_run(
        {
            "file": input_path.name,
            "course": course_id,
            "audio_min": round(duration / 60, 1),
            "source_mb": source_megabytes,
            "transcribe_s": round(elapsed, 1),
            "speed_x": round(duration / elapsed, 1) if elapsed > 0 else None,
            "recheck_s": round(recheck_seconds, 1),
            "recheck_n": n_rechecked,
            "upload_s": round(upload_seconds, 1),
            "uploaded": upload,
            "ok": ok,
        }
    )

    send_notification(
        "전사 완료" if ok else "업로드 실패", f"{input_path.name} — {audio_note}"
    )
    return ok, audio_note


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = sys.argv[1:]
    flags = {
        "--dump-json": False,
        "--no-upload": False,
        "--keep-audio": False,
        "--upload-only": False,
        "--recheck": False,
        "--carry-context": False,
        "--seed-glossary": False,
    }
    for flag in flags:
        if flag in args:
            flags[flag] = True
    args = [a for a in args if a not in flags]
    if flags["--upload-only"] and flags["--no-upload"]:
        print("Error: --upload-only와 --no-upload는 같이 쓸 수 없습니다.")
        sys.exit(1)
    if not args:
        print(
            "Usage: python3 transcribe_whisper.py "
            "[--dump-json] [--no-upload] [--keep-audio] [--upload-only] [--recheck] "
            "[--carry-context] [--seed-glossary] "
            "<audio.m4a> [audio2.m4a ...]"
        )
        sys.exit(1)

    # ffmpeg, ffprobe는 전사에만 필요 (업로드용 gws/notebooklm은 선택 — 없으면 그 단계만 실패로 보고)
    if not flags["--upload-only"]:
        check_dependencies()
    if not flags["--no-upload"]:
        for tool in ["gws", "notebooklm"]:
            if not shutil.which(tool):
                print(
                    f"⚠️  '{tool}' 없음 — 해당 업로드는 실패로 기록되고 원본은 삭제되지 않습니다."
                )

    input_paths = []
    for arg in args:
        p = Path(arg).expanduser()
        if p.is_symlink():
            # resolve()가 링크를 따라가므로, 삭제 단계에서 원본을 지울 위험이 있다
            print(f"심볼릭 링크는 처리하지 않습니다: {p}")
            sys.exit(1)
        p = p.resolve()
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)
        input_paths.append(p)

    print(f"\n🗂  Files to process: {len(input_paths)}")

    # Select language once for the entire batch
    language = None if flags["--upload-only"] else choose_language()

    success_count = 0
    fail_count = 0
    audio_notes = []

    for i, input_path in enumerate(input_paths):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(input_paths)}] {input_path.name}")
        print("=" * 60)

        if flags["--upload-only"]:
            output_path, course_id = resolve_output_path(input_path, create_dir=False)
            keep = flags["--keep-audio"]
            if not output_path.exists():
                # 전사문이 없는 파일을 지우면 원본과 전사가 함께 사라진다
                print(f"⚠️  전사문 없음({output_path.name}) — 원본은 삭제하지 않습니다.")
                keep = True
            success, note = run_dispatch(input_path, course_id, keep_audio=keep)
            send_notification(
                "업로드 완료" if success else "업로드 실패",
                f"{input_path.name} — {note}",
            )
        else:
            file_lang = (
                choose_language(input_path.name) if language == "PER_FILE" else language
            )
            success, note = transcribe_file(
                input_path,
                language=file_lang,
                dump_json=flags["--dump-json"],
                upload=not flags["--no-upload"],
                keep_audio=flags["--keep-audio"],
                recheck=flags["--recheck"],
                carry_context=flags["--carry-context"],
                seed_glossary=flags["--seed-glossary"],
            )

        audio_notes.append((input_path.name, note))
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"🏁 Processing Finished. Success: {success_count}, Fail: {fail_count}")
    for name, note in audio_notes:
        print(f"   • {name} → {note}")
    print(f"{'=' * 60}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
