# 로컬 Qwen과 QLoRA

## 실행

```bash
python3 /Users/mh/repository/adaptive-aac/setup_local.py --serve
```

Apple Silicon arm64 Mac, Python 3.11 이상, 패키지·공개 모델 다운로드와 MLX 실행이 가능한 환경이 필요합니다. 설치가 실패하면 일반 Mac 터미널에서 재실행하세요. 앱 내 명령 실행 환경은 네트워크·GPU·포트 권한이 제한될 수 있습니다.

`No module named aac`가 나오면 작업 폴더가 잘못된 것입니다. `python3 /Users/mh/repository/adaptive-aac/run.py`는 별도의 `cd`가 필요하지 않습니다.

## 구성

- 모델: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.
- 학습 런타임: `mlx-lm[train]==0.32.1`. 실제 설치된 하위 의존성은 각 실행의 `requirements-lock.txt`에 기록합니다.
- 모델 리비전: 다운로드 시 커밋 SHA를 조회하고 그 SHA로 다운로드합니다.
- 학습: QLoRA, 마지막 4개 레이어, 배치 1, 학습률 0.0001, 기본 80 iterations, prompt masking, gradient checkpointing.
- 입력 최대 2048 토큰. 넘는 예제는 조용히 자르지 않고 중단합니다.
- 추론: greedy sampling, 최대 생성 96 토큰, 별도 Python 워커에 모델을 한 번 로드. 서버가 워커를 재사용하고 요청을 직렬화합니다.
- 추론에 추가 HTTP 서버나 포트가 필요하지 않습니다. stdin/stdout JSON으로 통신하며 모델 워커는 Hugging Face offline 모드로 실행합니다.
- 추론 25초 초과/동시 요청 대기 1초 초과/파싱 오류/잘못된 어휘는 규칙 추천으로 복구합니다. 브라우저 추천 대기는 35초입니다.

출력의 `ids`는 고정 어휘, 요청 후보, 이미 선택한 그림 제외, 중복 제외, 개수 제한을 통과해야 합니다. 학습 지표는 규칙 fallback으로 보정하지 않은 **모델 자체 생성**으로 계산합니다.

## 데이터

기본 합성 데이터는 어휘의 먹기·마시기·가기·요청하기 조합과 더/거절, 통증, 휴식, 기본 응답 템플릿입니다. `source-synthetic.json`에 생성된 모든 예제를 남깁니다. 실제 개인 이력은 자동 학습하지 않습니다.

먼저 앞 두 그림으로 의도 묶음을 만들고, 같은 묶음의 상황 변화와 부정/추가 변형을 같은 split에 넣습니다. 이후 접두 선택 예제로 확장합니다. 동일한 입력 프롬프트가 여러 split에 있으면 학습 → 검증 → 평가 순으로 한 split에만 남깁니다. 동일 접두사에 여러 실제 다음 선택이 있으면 최대 6개의 타깃 ID로 합칩니다. 타깃 순서는 합성 자료 순서이므로 실제 선호나 빈도 추정치가 아닙니다.

기본 데이터의 현재 분리는 학습 50개, 검증 12개, 평가 10개 입력입니다. 이 자료는 서로 연관된 규칙으로 생성한 작은 개발 세트입니다. 프롬프트 중복을 제거해도 사용자/시나리오 일반화 성능의 독립적 근거는 아닙니다. 실제 사용자별 분리는 이 도구가 자동으로 해결하지 않습니다.

## 결과와 실패 확인

`var/local-runs/<실행 ID>/`:

| 파일 | 내용 |
| --- | --- |
| `setup-status.json` | 설치와 실행 단계, 성공/실패 |
| `data/manifest.json` | 데이터 종류, split 개수, 중복 제거, SHA-256 |
| `training-status.json` | 실제 학습 시작/완료 여부, 실패 단계, 모델 리비전 |
| `training.log` | MLX가 출력한 loss·학습 진행 |
| `before.json`, `after.json` | 학습 전후 유효 JSON 비율, hit@6, 추론 시간, 원시 출력 |
| `adapter/adapters.safetensors` | 실제 학습한 LoRA 가중치 |
| `requirements-lock.txt` | 해당 실행의 설치 패키지 버전 |

`setup-status.json`이 `failed`이면 실행은 완료되지 않은 것입니다. 데이터 파일이 생성된 것과 모델이 학습된 것은 다릅니다. 학습 subprocess가 성공하고 어댑터 파일이 존재한 뒤에만 `training_completed: true`가 기록됩니다.

학습 후 모델이 유효한 추천을 하나도 생성하지 못하면 활성화하지 않습니다. 일부 유효 출력을 냈다고 기존 모델보다 개선됐다고 가정하지 않습니다. `before.json`과 `after.json`의 실패 사례·지연·적중을 직접 확인하세요. 현재 명령은 전후 비교에 test split을 사용하고 유효 출력 여부를 활성화 조건으로도 사용하므로, 엄밀한 최종 성능 보고에는 별도의 손대지 않은 holdout 세트가 필요합니다.

성공하면 `var/active-model.json`을 원자적으로 바꾸고 다음 앱 실행에 해당 어댑터를 로드합니다. `.env`는 덮어쓰지 않습니다. 과거 실행 디렉터리와 가중치는 보존합니다.

## 수동 설정

`.env`의 기본값 `AAC_LOCAL_MODEL=auto`는 성공적으로 활성화한 모델을 읽습니다. 직접 경로를 지정할 수도 있습니다.

```dotenv
AAC_LOCAL_MODEL=/absolute/path/to/downloaded/model
AAC_LOCAL_ADAPTER=/absolute/path/to/run/adapter
```

`AAC_LOCAL_MODEL=off`는 로컬 추론을 끕니다. 클라우드 옵션이 꺼져 있으면 규칙 추천을 사용합니다. 로컬 학습된 모델과 학습 전 베이스 모델을 비교하려면 `AAC_LOCAL_ADAPTER`를 비워 베이스 모델만 사용하세요. 로컬 모델에는 OpenAI API 키가 필요하지 않습니다.

## 참고

- [모델 카드](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit)
- [MLX LM LoRA 공식 문서](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX LM Python 추론 API](https://github.com/ml-explore/mlx-lm#python-api)
