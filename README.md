# 온말 · Adaptive AAC

그림을 선택하면 의도에 맞는 다음 그림을 추천하고, 선택한 개념을 한국어 문장과 음성으로 전하는 **로컬 연구용 AAC 프로토타입**입니다.

원안: [Personalized AAC with LLM-Based Adaptive Prediction](https://docs.google.com/document/d/1OREAwXGHU4B_dt4vumBLO0tI30KbPHGEbdKrIRYUg6g/edit?tab=t.0). 제안서의 한 학기 프로토타입 범위를 기준으로 구현했습니다. 임상 검증이나 사용자 대상 효과 검증은 수행하지 않았습니다.

## 바로 실행

Python **3.11 이상**만 있으면 됩니다. 런타임 패키지, Node.js, 빌드, API 키가 필요하지 않습니다.

```bash
cd /Users/mh/repository/adaptive-aac
python3 -m aac
```

브라우저에서 **http://localhost:8765**를 여세요. 종료는 `Ctrl+C`입니다.

1. `먹고 싶어요` → `빵`을 고릅니다.
2. **문장으로 만들기**를 눌러 `빵을 먹고 싶어요.`를 확인합니다.
3. 필요하면 문장을 직접 고친 뒤 **이 문장으로 전달하기**를 누릅니다.
4. 큰 글자로 보여 주거나 **소리로 읽기**를 누릅니다. 음성은 자동 재생되지 않습니다.

추천에 원하는 그림이 없으면 **다른 그림 추천**, 분류 탭, **전체 그림 보기**를 사용하세요. `처음부터`로 새 메시지를 시작하면 기본 그림판 모드로 전환할 수 있습니다.

포트 충돌 시 `AAC_PORT=8766 python3 -m aac`로 실행합니다. 프로그램은 `127.0.0.1`에만 바인딩하므로 다른 기기에서는 접속할 수 없습니다.

## 구현 기능

- 7개 분류, 47개 고정 어휘와 한국어 이름, 그림 버튼. 초기 그림은 시제품용 시스템 이모지입니다.
- 선택 순서·상황·확정한 이전 메시지를 이용한 다음 그림 6개 추천.
- 추천 새로고침, 전체 어휘 검색, 분류 탐색, 개별 취소·되돌리기·초기화.
- 한국어 조사와 부정·추가 요청을 반영한 문장 조합, 사용자의 문장 수정·확정.
- 큰 글자 전달 화면과 브라우저 SpeechSynthesis 기반 읽기·중지.
- SQLite 개인화 이력, 이전 메시지 다시 표시, JSON 내보내기·삭제.
- 동적 추천 / 고정 그림판 비교, 확정한 메시지별 선택 동작 수·작성 시간 기록.
- 선택적 OpenAI LLM 추천, 제한된 JSON Schema 출력, 잘못된 결과·시간 초과 시 기본 추천.
- 합성 시나리오 평가와 다음 그림 예측용 supervised JSONL 준비 도구.

## LLM 추천 연결

```bash
cp .env.example .env
```

`.env`에 본인의 키와 사용할 모델 ID를 넣고 서버를 다시 실행하세요.

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
AAC_PORT=8765
AAC_DB_PATH=var/aac.sqlite3
```

그 다음 앱의 **설정 → AI 맞춤 추천 사용**을 켭니다. 키만 설정했을 때는 외부 호출을 하지 않습니다. 기본 예시 모델 외에도 Responses API와 해당 JSON Schema를 지원하는 모델 ID를 지정할 수 있습니다. 계정에서 해당 모델을 사용할 수 있어야 합니다.

LLM에 보내는 데이터는 현재 선택한 그림, 사용자가 고른 상황, 최근 확정한 그림 순서 최대 12건, 허용 어휘입니다. 이름·사용자가 직접 수정한 문장·프로필 토큰은 보내지 않습니다. 키는 서버에서만 읽고 `.env`는 Git에서 제외합니다. API 요청은 `store: false`를 사용하며 공급자의 데이터 처리 정책이 별도로 적용됩니다.

추천 결과는 서버가 허용 ID·중복·개수·이미 선택한 그림·제외한 그림을 다시 검증합니다. 문장은 서버가 선택 개념으로 만든 후보 중 하나만 모델이 고를 수 있게 하여 추가 의미를 생성하지 않도록 제한했습니다. 임의의 복합 개념은 원래 그림 이름을 그대로 보존하며 사용자가 문장을 수정할 수 있습니다.

구현 근거: [OpenAI Structured Outputs 공식 문서](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses). 실제 유료 API 호출은 구현 검증 과정에서 수행하지 않았으며, 테스트는 모의 응답을 사용합니다.

## 테스트와 평가

추가 패키지 없이 핵심 테스트 실행:

```bash
python3 -m unittest discover -s tests -v
python3 -m scripts.evaluate --output artifacts/evaluation.json
python3 -m scripts.prepare_training data/scenarios.json --output artifacts/synthetic-training.jsonl
```

테스트는 추천 범위·개인화·재추천·부정 표현·잘못된 모델 응답·SQLite 저장·프로필 분리·중복 저장 방지·HTTP 입력 검증·외부 Origin 차단을 확인합니다. 실제 HTTP 처리기를 메모리 스트림으로 실행하므로 테스트에 포트가 필요하지 않습니다. `pytest`를 선호하면 `requirements-dev.txt`로 설치한 뒤 동일 테스트를 실행할 수 있습니다.

선택적 화면 테스트는 Node 20+와 설치된 Chromium이 필요합니다. 런타임 앱은 Node를 사용하지 않습니다.

```bash
CHROMIUM_PATH=/absolute/path/to/chromium node scripts/browser-test.mjs
```

브라우저 테스트는 CDP 파이프와 임시 SQLite DB를 사용하며, 실제 HTTP 처리기를 통해 화면 선택→문장 생성→확정→이력, 전체 어휘 검색, 취소, 기본 그림판, 모바일 너비, 설정과 삭제를 검사하고 `artifacts/`에 화면을 저장합니다. Python 경로는 `PYTHON`으로 지정할 수 있습니다. 브라우저 실행 권한이 제한된 환경에서는 실행되지 않을 수 있습니다.

`data/scenarios.json`은 손으로 만든 **개발용 합성 데이터 16건**입니다. 평가 수치는 6개 그림 단위 탐색의 시뮬레이션이며 실제 사용자의 성능 향상을 의미하지 않습니다. 사용자 평가 설계와 계산 가정은 [평가·학습 안내](docs/evaluation.md)를 확인하세요.

## 학습 데이터와 파인튜닝

설정에서 내보낸 `aac-interactions.json`을 검토한 뒤 변환합니다.

```bash
python3 -m scripts.prepare_training /path/to/aac-interactions.json --output var/training.jsonl
```

도구는 `(이전 선택 그림 → 실제 다음 선택 그림)` 예제를 만들고 같은 예제는 중복 제거합니다. 이름·프로필 ID·자유 입력 문장은 학습 예제에 넣지 않습니다. 동일 문장 중복에 따른 가중치 학습이 필요하면 별도로 표본 설계를 해야 합니다.

**실제 모델 학습은 아직 실행하지 않았습니다.** 제안서의 2단계 파인튜닝에는 검토·동의된 실제 상호작용 데이터, 사용자/세션 단위 학습·검증 분리, 사용할 모델과 학습 환경 결정이 필요합니다. 합성 데이터는 형식 확인용이며 학습 효과의 근거가 아닙니다. 학습 후 Responses API 호환 모델 ID를 `OPENAI_MODEL`에 설정해 동일 추천 경로로 비교할 수 있습니다. 데이터 업로드나 유료 학습 작업은 이 도구가 자동 실행하지 않습니다.

## 구조

```text
aac/                 Python API, 추천, 문장, LLM 어댑터, SQLite
web/                 한국어 화면 (HTML / CSS / JavaScript)
data/scenarios.json  합성 평가 시나리오
scripts/             평가, 학습 데이터 준비, 선택적 브라우저 테스트
tests/               도메인·API 자동 테스트
docs/                기능 대응표, 평가 설계
var/                 로컬 DB / 사용자 데이터 (Git 제외)
artifacts/           생성한 평가 / 학습 파일 / 화면 (Git 제외)
```

설계 설명과 API 계약은 [구현 안내](docs/implementation.md)에 있습니다.

## 현재 범위

한국어 단일 언어, 단일 로컬 컴퓨터 사용을 전제로 합니다. 브라우저별 익명 프로필 토큰으로 기록이 분리되며 서비스용 계정 인증은 없습니다. localhost와 127.0.0.1은 브라우저 저장 공간이 다르므로 같은 주소로 접속하세요. 브라우저 저장 공간을 삭제하면 기존 프로필 기록에 UI로 다시 접근할 수 없습니다. `var/aac.sqlite3`를 백업하면 서버 데이터는 보존됩니다.

실제 배포 전에는 사용자·보호자와의 어휘/그림 검증, 정식 AAC 심볼 라이선스 검토, 접근성 사용자 테스트, 인증·운영 서버 설계가 필요합니다. 시스템 이모지는 기기마다 모양이 다르며 임상용 AAC 심볼을 대체하는 검증된 자료가 아닙니다. TTS는 기기의 한국어 음성 설치 및 브라우저 지원에 따라 달라지고, 해당 기기의 음성 서비스 정책을 따릅니다.
