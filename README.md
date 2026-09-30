# whisper-lecture

강의 전사 파이프라인이 **얼마나 틀리는지 숫자로 재는** 레포입니다. 모델을 바꾸거나
학습시키는 곳이 아닙니다 — 자(尺)를 먼저 만들고, 숫자를 보고 다음 수단을 고릅니다.

측정 대상은 두 세대입니다.

- **(A) 26-1 출고본** — 2026-08-26 설정 변경 *이전*에 만들어져 이미 노트 작성에 쓰인 전사문
- **(B) 현재 파이프라인** — 오늘 다시 돌렸을 때 나오는 결과

## 왜 두 가지 파이썬을 쓰나

| 무엇 | 실행 | 이유 |
|---|---|---|
| `transcribe_whisper.py` (전사) | 전역 `python3` | `mlx-whisper` 가 `pip --break-system-packages` 로 전역에 깔려 있습니다. venv 에 넣지 않습니다 |
| `scripts/*.py` (측정) | `uv run` | 런타임 의존성은 `jiwer` 하나뿐입니다 |

`pyproject.toml` 에 `mlx-whisper` 를 추가하지 마세요. 전사 스크립트는 이 레포에 있지만
레포의 venv 와는 무관하게 돌아갑니다 (ruff 검사에서도 제외되어 있습니다).

## 지표

모두 `unicodedata.normalize("NFC", …)` 이후에 계산합니다. macOS·iCloud 가 한글을 NFD 로
저장하기 때문에, 정규화하지 않으면 같은 글자가 자모 단위로 어긋나 편집거리가 부풀려집니다.

- **CER** — 공백·문장부호를 버리고 `jiwer.cer`. 한국어는 띄어쓰기 차이가 노이즈라 WER 대신 CER
- **용어 오인식률** — 정답본에 나온 용어 중 가설이 틀린 비율. CER 6 % 여도 핵심 용어를 전부
  깨뜨릴 수 있고, 실제 불만은 용어 쪽입니다
- **누락률** — 정답본 어절 중 가설에 대응이 없는 비율 (빠른 발화 누락)

## 절차

```bash
# 1. Drive 에서 원본 1개 받기 (stem 은 과목 키워드를 피한 ASCII)
uv run scripts/fetch.py s1-261 <folder_id> "260512 컴퓨터네트워크w11-1.m4a"

# 2. 분 경계에서 5분 발췌 (ffprobe 로 300초 확인, 어긋나면 재인코딩)
uv run scripts/make_excerpt.py s1-261 48

# 3. 현재 파이프라인으로 재전사 — 업로드·삭제 금지 플래그는 필수 (사이드카는 data/review/<stem>.review.jsonl)
cp data/excerpt/s1-261.excerpt.m4a data/hyp/s1-261.current.m4a
python3 transcribe_whisper.py --no-upload --keep-audio --dump-json \
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

플래그가 없는 구간도 읽어야 합니다. `avg_logprob`·`no_speech_prob` 는 세그먼트가 아니라
30초 디코딩 창 단위 값이고, 확신에 찬 오인식(예: "스택 카나리" → "스택 하나리",
`avg_logprob` -0.08)은 플래그가 전혀 붙지 않습니다.

## 정기 점검 — 용어 채굴 + 비용 (2주마다)

2주마다 돌리는 절차입니다. GPU 는 쓰지 않습니다.

```bash
cd whisper-lecture
V="<WIKI_VAULT>/10. Raw Sources"

# 1. 최근 14일 전사문에서 깨진 용어 후보 (과목별로 그 과목 용어집 기준)
for d in "$V"/201.*/; do
  uv run scripts/mine_variants.py "$d" --course "$(basename "$d")" --since 14
done

# 2. 같은 기간의 실행 비용 (GPU 시간 = 전사 초, 네트워크 = 업로드 초)
uv run scripts/run_report.py
```

후보의 절반가량은 오탐입니다(2026-09-22 실측 48%). **귀가 아니라 강의자료로 판정합니다** —
볼트 `50. Assets/<과목>/*.pdf` 를 `pdftotext` 로 뽑아 문자열 대조하고, 애매하면 NotebookLM 에
원문 인용을 요구합니다. **NLM 답은 그 인용이 함께 돌려준 근거 조각(`references[].cited_text`)에
실제로 있을 때만 씁니다** — NLM 은 시각이 아니라 내용으로 원문을 찾으므로 "몇 분 몇 초에 뭐라고 했나"
는 묻지 말고 "이런 내용이 나오는 대목"으로 묻습니다(2026-09-23 실측: 내용 질문 인용 81% 일치,
시각 질문 0/3). whisper 와 NLM 이 같은 답인데 강의자료와 다르면 둘 다 틀린 것일 수 있으니 소유자가
정합니다. 확정한 것만 `glossary/terms.tsv` **오답 열**에, 그 후보가 나온 **과목 범위**로
넣습니다(`aliases.tsv` 아님). 그다음 `uv run scripts/fix_terms.py --write <폴더>` 로 볼트에 소급합니다.

**언제 그만두나.** 정해진 횟수는 없습니다. 오답은 새 단원과 함께 들어오기 때문입니다 —
2026-09-22 새 강의 11개에서 기존 오답 규칙이 걸린 곳은 1곳뿐이었습니다. 회차마다 확정 건수를
`local/verify-queue.md` 에 적고, 확정·기각(이유)·미검토 후보 전체는 `local/mining/<날짜>-roundN.md` 에
남깁니다 — **같은 후보가 다음 회차에 또 나오면 그 파일의 판단부터 봅니다**(강의 문장이 들어 있어 레포 밖).
그리고 **확정 0건이 두 회차 연속**이면 은퇴합니다. 학기가 끝나 새 강의가
없어져도 은퇴입니다.

## 레포 규칙

- `data/` 는 전부 gitignore 입니다. 강의 오디오·전사문에는 다른 사람의 목소리와 이름이
  들어 있어 **어떤 것도 커밋하지 않습니다**
- `results/` 는 숫자만 둡니다. 전사문을 인용하지 않습니다
- `glossary/terms.tsv` 는 추적합니다 (용어 목록에는 개인 정보가 없습니다)
