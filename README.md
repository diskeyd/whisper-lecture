# whisper-lecture

강의 전사 파이프라인이 **얼마나 틀리는지 숫자로 재는** 레포입니다. 모델을 바꾸거나
학습시키는 곳이 아닙니다 — 자(尺)를 먼저 만들고, 숫자를 보고 다음 수단을 고릅니다.

측정 대상은 두 세대입니다.

- **(A) 26-1 출고본** — 2026-08-26 설정 변경 *이전*에 만들어져 이미 노트 작성에 쓰인 전사문
- **(B) 현재 파이프라인** — 오늘 다시 돌렸을 때 나오는 결과

## 현재 결론 (2026-10-13)

- **엔진**: mlx-whisper large-v3 그리디 + 반복 필터 + 용어 치환 (소유자 결정 2026-10-13)
- **녹음**: 아이폰 한 대, 음성분리 끔, 자동 모드 (2026-09-30 판정)
- **다음 레버**: 용어 채굴·치환 강화 (LLM 후편집은 강의 정리 스킬과 중복이라 쓰지 않음)

근거 수치는 [`docs/summary.md`](docs/summary.md), 측정 원기록은 `results/2026-09-baseline.md`.

## 왜 두 가지 파이썬을 쓰나

| 무엇 | 실행 | 이유 |
|---|---|---|
| `transcribe_whisper.py` (전사) | brew `python3.14` | `mlx-whisper` 가 brew python@3.14 에 `pip --break-system-packages` 로 깔려 있습니다 (2026-10-10 부터 기본 `python3` 은 3.15 라 패키지가 없습니다). venv 에 넣지 않습니다. shebang 이 brew python3.14 로 고정돼 있어 `./transcribe_whisper.py` 로도 돌아갑니다 |
| `scripts/clip_check.py`, `scripts/recheck_replay.py` | brew `python3.14` | 두 스크립트는 `mlx_whisper` 와 `transcribe_whisper` 를 임포트하므로 `uv run` 으로는 안 됩니다 |
| 나머지 `scripts/*.py` (측정) | `uv run` | 런타임 의존성은 `jiwer`, `rapidfuzz`(채굴의 유사도). 개발용 `pytest`·`ruff` 는 dev 그룹 |

`pyproject.toml` 에 `mlx-whisper` 를 추가하지 마세요. 전사 스크립트는 이 레포에 있지만
레포의 venv 와는 무관하게 돌아갑니다 (ruff 검사에서도 제외되어 있습니다).

## 설정 — 이 컴퓨터에만 있는 값

볼트 폴더와 Drive 폴더 id 는 코드에 두지 않습니다. 환경변수나 `local/config.toml` 에서 읽습니다.
`config.example.toml` 을 `local/config.toml` 로 복사해 채웁니다 (`local/` 은 gitignore 입니다).
둘 다 있으면 환경변수가 이깁니다.

| 환경변수 | `config.toml` 키 | 쓰임 |
|---|---|---|
| `WHISPER_VAULT` | `vault` | 메인 볼트. `30. Permanent Notes/` 의 Subject 노트에서 과목별 NotebookLM 노트북 id 를 읽습니다 |
| `WHISPER_WIKI_VAULT` | `wiki_vault` | 전사문을 쌓는 볼트. `10. Raw Sources/<과목 ID>/` 아래에 저장합니다 (`scripts/slice_transcript.py` 도 여기서 읽습니다) |
| `WHISPER_DRIVE_FOLDER_ID` | `drive_folder_id` | 녹음을 올릴 Drive 폴더 id. 업로드하는 실행에만 필요합니다 |

측정 도구(`cer.py`, `term_error.py` 등)와 테스트는 설정 없이 돕니다. 전사 스크립트는 값이 필요한 자리에서
변수 이름을 알려 주고 멈춥니다. 업로드하는 실행은 시작할 때 `WHISPER_VAULT` 를 확인합니다. Drive id 만 없으면
그 업로드가 실패로 기록되고 원본은 지워지지 않습니다.

### `transcribe_whisper.py` 플래그

| 플래그 | 하는 일 |
|---|---|
| (없음) | 전사 → 볼트 저장 → Drive·NotebookLM 업로드 → 둘 다 성공하면 원본 삭제 |
| `--no-upload` | 전사만. 업로드도 삭제도 하지 않습니다 (측정할 때는 이것만으로 충분) |
| `--keep-audio` | 업로드는 하되 원본은 남깁니다 |
| `--upload-only` | 전사를 건너뛰고 업로드·삭제만 (재시도용, 파트 병합도 하지 않음) |
| `--dump-json` | 후처리 전 세그먼트를 `<stem>.whisper.json` 으로 남깁니다 |
| `--recheck` | 저신뢰 구간을 2차 디코딩해 사이드카에 `disagree` 로 대조 (기본 꺼짐) |
| `--carry-context`, `--seed-glossary` | 창 간 문맥 유지, 용어집 프롬프트 (실험용 — 둘을 함께 쓴 설정은 기각, `--seed-glossary` 단독은 미측정) |
| `--plan` | 과목·저장 폴더·업로드 여부만 JSON 줄로 출력. 오디오를 건드리지 않고 mlx 도 필요 없습니다 |
| `--no-merge` | 나뉜 녹음도 병합하지 않고 파일마다 따로 전사합니다 (`제목(1).md`, `제목(2).md`). `--plan` 과 함께 쓰면 파일마다 한 줄 |

음성메모는 앱에서 `~/Downloads` 로 끌어다 놓고 경로를 넘깁니다 (터미널에 전체 디스크 접근 권한을 주지 않기 위해
음성메모 목록을 자동으로 읽지 않습니다). 녹음이 `제목(1)`·`제목(2)` 로 나뉘었으면
함께 넘기면 `제목.m4a` 로 병합해 한 번만 전사합니다. 파트는 **병합본이 지워졌을 때만**(Drive 업로드 성공 + NotebookLM
준비 완료) 함께 지웁니다 — `--no-upload`·`--keep-audio` 이거나 노트북이 없는 과목이면 병합본과 파트가 모두 남습니다.
한쪽만 넘기면 같은 폴더의 나머지 파트를 알려 주고 진행 여부를 묻습니다. `--no-merge` 를 주면 병합도 이 확인도 없이
파트마다 따로 전사하고(`제목(1).md`), 지우는 규칙은 파일 하나일 때와 같습니다(각 파트는 자기만 지움). 자동 호출(디스패치 봇 등)은 항상
`--no-merge` 로 돌립니다. 음성메모 앱의 녹음은 직접 지웁니다.

## 지표

모두 `unicodedata.normalize("NFC", …)` 이후에 계산합니다. macOS·iCloud 가 한글을 NFD 로
저장하기 때문에, 정규화하지 않으면 같은 글자가 자모 단위로 어긋나 편집거리가 부풀려집니다.

- **CER** — 공백·문장부호를 버리고 `jiwer.cer`. 한국어는 띄어쓰기 차이가 노이즈라 WER 대신 CER.
  `cer.py` 는 두 줄을 냅니다: **적힌 대로**, 그리고 `glossary/aliases.tsv` 의 표기 변형(`라이센스`/`라이선스` 등)을
  양쪽에서 접은 **표기 통일**
- **용어 오인식률** — 정답본에 나온 용어 중 가설이 틀린 비율. CER 6 % 여도 핵심 용어를 전부
  깨뜨릴 수 있고, 실제 불만은 용어 쪽입니다. 세 번째 인자로 과목 ID 를 주면 그 과목 용어만 셉니다.
  별칭은 접어서 맞은 것으로 치되 `notonly` 표시가 붙은 별칭은 틀린 것으로 세고, 긴 용어 안에 든 짧은 용어
  (`완전 이진트리` 안의 `이진트리`)는 한 번만 셉니다
- **누락률** — 정답본 어절 중 가설에 대응이 없는 비율 (빠른 발화 누락). 붙어서 나온 어절도 누락으로
  세므로 **상한값**입니다

정답본·가설 모두 `textnorm.load_body` 가 `**m:ss - m:ss**` 블록 머리와 `<…>` 교정 메모를 걷어 낸 뒤 잽니다.

## 절차

```bash
# 1. Drive 에서 원본 1개 받기 (stem 은 과목 키워드를 피한 ASCII, folder_id 는 위 설정의 Drive 폴더 id)
uv run scripts/fetch.py s1-261 <folder_id> "260512 컴퓨터네트워크w11-1.m4a"

# 2. 분 경계에서 5분 발췌 (ffprobe 로 300초 확인, 어긋나면 재인코딩)
uv run scripts/make_excerpt.py s1-261 48

# 3. 현재 파이프라인으로 재전사 — --no-upload 필수 (사이드카는 data/review/s1-261.current.review.jsonl)
cp data/excerpt/s1-261.excerpt.m4a data/hyp/s1-261.current.m4a
python3.14 transcribe_whisper.py --no-upload --dump-json \
    data/hyp/s1-261.current.m4a </dev/null

# 4. 26-1 샘플은 출고본의 같은 구간도 잘라낸다 (볼트는 읽기 전용)
uv run scripts/slice_transcript.py s1-261 "201.66 컴퓨터네트워크" "260512 컴퓨터네트워크w11-1.md"

# 5. 교정 시트 만들기
uv run scripts/review_sheet.py data/review/s1-261.current.review.jsonl

# 6. 정답본이 생긴 뒤 — 측정
uv run scripts/cer.py data/gold/s1-261.md data/hyp/s1-261.current.md
uv run scripts/term_error.py data/gold/s1-261.md data/hyp/s1-261.current.md "201.66 컴퓨터네트워크"
```

`stem` 에 과목 키워드(`컴퓨터네트워크`, `알고리즘` 등)가 들어가면 전사 스크립트가 출력을
볼트의 과목 폴더로 보내 **출고본을 덮어씁니다**. `fetch.py` 가 이 조건을 검사해서 막습니다.

## 정답본 만들기 — 사람이 합니다

1. `cp data/hyp/<stem>.current.md data/gold/<stem>.md` — **이 파일을 고칩니다.**
   시트(`data/hyp/<stem>.sheet.md`)는 "어디를 주의해서 들을지" 안내용입니다.
   시트를 그대로 정답본으로 쓰면 세그먼트마다 붙은 `**3:20**` 표시가 따라 들어갑니다
   (`load_body` 가 걸러내지만, 원본 형식을 유지하는 편이 안전합니다)
2. 오디오(`data/excerpt/<stem>.excerpt.m4a`)를 들으며 **틀린 곳만** 고칩니다 (발췌당 20~30분)
3. 깨진 기술 용어는 `glossary/terms.tsv` 의 오답 열에 Whisper 가 쓴 표기를 그대로 적습니다

정답본을 LLM 으로 만들거나 초안을 LLM 으로 미리 고치지 마세요. 모델이 만든 정답으로 재면
정확도가 아니라 **모델끼리의 일치도**를 재게 됩니다.

정답본은 출발한 초안 쪽에 유리합니다(고치는 사람은 들리는 대로 맞으면 그대로 둡니다). s1~s3 는 whisper 초안,
s4-oss·s5-crypto 는 Qwen3-ASR 초안에서 출발했습니다. 그래서 **엔진끼리 비교할 때는 정답본 CER 대신**
`scripts/judge_sheet.py` 로 두 전사가 다른 구간만 들어서 가립니다.

플래그가 없는 구간도 읽어야 합니다. `avg_logprob`·`no_speech_prob` 는 세그먼트가 아니라
30초 디코딩 창 단위 값이고, 확신에 찬 오인식(예: "스택 카나리" → "스택 하나리",
`avg_logprob` -0.08)은 플래그가 전혀 붙지 않습니다.

## 정기 점검 — 용어 채굴 + 비용 (2주마다)

2주마다 돌리는 절차입니다. GPU 는 쓰지 않습니다.

```bash
cd whisper-lecture
V="<wiki_vault>/10. Raw Sources"   # 위 설정의 wiki_vault

# 1. 최근 14일 전사문에서 깨진 용어 후보 (과목별로 그 과목 용어집 기준)
for d in "$V"/201.*/; do
  uv run scripts/mine_variants.py "$d" --course "$(basename "$d")" --since 14
done

# 2. 실행 비용 (GPU 시간 = 전사 초, 네트워크 = 업로드 초) — 전사 스크립트가 쌓는 data/run-log.jsonl 기준.
#    기간 필터는 없습니다: 인자 없이 전체 날짜별, 또는 하루만 YYYY-MM-DD
uv run scripts/run_report.py
```

후보의 절반가량은 오탐입니다. **귀가 아니라 강의자료로 판정합니다** —
볼트 `50. Assets/<과목>/*.pdf` 를 `pdftotext` 로 뽑아 문자열 대조하고, 애매하면 NotebookLM 에
원문 인용을 요구합니다. **NLM 답은 그 인용이 함께 돌려준 근거 조각(`references[].cited_text`)에
실제로 있을 때만 씁니다** — NLM 은 시각이 아니라 내용으로 원문을 찾으므로 "몇 분 몇 초에 뭐라고 했나"
는 묻지 말고 "이런 내용이 나오는 대목"으로 묻습니다(시각 질문은 실측에서 맞힌 적이 없습니다). whisper 와 NLM 이 같은 답인데 강의자료와 다르면 둘 다 틀린 것일 수 있으니 소유자가
정합니다. 확정한 것만 `glossary/terms.tsv` **오답 열**에, 그 후보가 나온 **과목 범위**로
넣습니다(`aliases.tsv` 아님). 그다음 `uv run scripts/fix_terms.py --write <폴더>` 로 볼트에 소급합니다.
`fix_terms.py` 는 과목 범위를 보지 않고 전체 용어집을 폴더에 적용하므로, **폴더는 과목별로 넘기고 `--write` 전에
모의 실행으로** 다른 과목 표기(예: 201.69 `하이라이키컬`, 조사가 붙은 `위험모델`)가 걸리지 않았는지 확인합니다.
가드가 건너뛴 곳은 모의 실행 출력의 「건너뜀」 아래 `보류` 로 나오니 문맥을 읽고 직접 처리합니다.
`fix_terms.py` 는 `aliases.tsv` 의 표기 변형(`notonly` 포함)도 함께 적용하므로, `--write` 하면 볼트의 별칭 표기도 바뀝니다.

**언제 그만두나.** 정해진 횟수는 없습니다. 오답은 새 단원과 함께 들어오기 때문입니다 —
2026-09-22 새 강의 11개에서 기존 오답 규칙이 걸린 곳은 1곳뿐이었습니다. 회차마다 확정 건수를
`local/verify-queue.md` 에 적고, 확정·기각(이유)·미검토 후보 전체는 `local/mining/<날짜>-roundN.md` 에
남깁니다 — **같은 후보가 다음 회차에 또 나오면 그 파일의 판단부터 봅니다**(강의 문장이 들어 있어 레포 밖).
그리고 **확정 0건이 두 회차 연속**이면 은퇴합니다. 학기가 끝나 새 강의가
없어져도 은퇴입니다.

## 레포 규칙

- `data/`·`local/` 은 전부 gitignore 입니다. 강의 오디오·전사문에는 다른 사람의 목소리와 이름이
  들어 있어 **어떤 것도 커밋하지 않습니다**. `local/` 에는 채굴 기록(`verify-queue.md`, `mining/`)처럼
  강의 문장이 들어간 메모를 둡니다
- `results/` 는 측정 원기록(날짜별 절), `docs/` 는 그것을 묶은 요약 보고서입니다. 둘 다 숫자만 두고
  전사문을 인용하지 않습니다 (예외: 오류 모양을 보여 주는 용어 몇 개)
- `glossary/terms.tsv`·`glossary/aliases.tsv` 는 추적합니다 (용어 목록에는 개인 정보가 없습니다)

## 보조 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/snr_probe.py <파일\|폴더> …` | 녹음을 말소리 대 바닥 소음(dB)으로 정렬 — 발췌 표본 고르기. 잔향은 재지 않음 |
| `scripts/judge_sheet.py make\|score <stem>…` | 두 전사가 다른 구간만 나란히 놓은 판정표를 만들고(`data/judge/`), 들은 판정으로 승률·CER 을 냄 — 정답본 편향 없이 엔진 비교 |
| `scripts/post_edit.py <전사문.md> ["과목 ID"]` | (측정 재현용 — 운영에서 쓰지 않음) 치환 뒤 남은 오인식을 Claude(`claude -p`, 기본 Sonnet low, 도구 없음)가 고침. 원본은 두고 `.post.md` 와 변경 이력 `.post.tsv` 를 씀. 블록 안에서 한 번만 나오는 조각만 바꿈 |
| `scripts/device_diff.py` | 같은 강의를 두 기기로 녹음한 전사의 차이 구간 목록 (기기 쌍 비교) |
| `python3.14 scripts/clip_check.py check\|recut <폴더>` | 검증 클립에 대상 단어가 실제로 들었는지 확인하고 다시 자름 |
| `python3.14 scripts/recheck_replay.py` | `--recheck` 단계만 다른 설정으로 다시 돌려 비교 |

## 테스트

```bash
uv run pytest
```

지표(`cer`, `term_error`, `textnorm`, 별칭, `replace_guarded`), `review_sheet`, `slice_transcript` 정렬,
`mine_variants`, `fetch`, `clip_check`, `device_diff`, `judge_sheet`, `post_edit` 의 수정 적용 규칙, 그리고 전사 스크립트의 업로드·삭제 조건, 사이드카 경로,
파트 묶기·병합, `--plan` 을 덮습니다(전사 스크립트 테스트는 `mlx_whisper` 를 스텁으로 바꿔 끼웁니다).
전사·반복 필터 본체, `fix_terms.py` CLI, `run_report.py`, `make_excerpt.py`, `snr_probe.py`, `recheck_replay.py` 는
테스트가 없습니다.
