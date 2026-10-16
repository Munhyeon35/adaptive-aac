# Windows·macOS·Linux의 로컬 Qwen

## 실행

```bash
git clone https://github.com/Munhyeon35/adaptive-aac.git
cd adaptive-aac
python setup_local.py --serve
```

Python 3.11 또는 3.12를 권장합니다. macOS·Linux에서는 `python3`, Windows에서는 `py -3.11`로 바꿔 실행해도 됩니다. Apple Silicon은 MLX, 다른 OS는 Transformers/PyTorch/PEFT를 자동 선택합니다. Windows·Linux는 x86-64를 우선 대상으로 합니다. Intel Mac은 PyTorch x86-64 휠이 제공되는 Python 3.11/3.12와 PyTorch 2.2.2를 사용합니다.

설치는 인터넷 연결과 충분한 디스크 공간이 필요합니다. CPU 경로는 약 1GB의 기본 모델을 내려받아 FP32로 로드하므로, MLX 4비트 경로보다 실행 메모리가 더 필요합니다. 학습 시간은 CPU/GPU에 따라 달라집니다. 앱 내 실행 도구가 GPU·네트워크·포트를 제한하면 일반 터미널에서 실행하세요. 사용자의 경로로 `python /path/to/adaptive-aac/run.py`를 실행하면 작업 폴더와 관계없이 시작합니다.

## 백엔드 선택과 CUDA

| 옵션 | 사용 방식 |
| --- | --- |
| 기본 `--backend auto` | Apple Silicon이면 MLX, 나머지는 Transformers |
| `--backend transformers --device cpu --training lora` | 모든 지원 OS에서 CPU 추론·LoRA 학습 |
| `--backend transformers --device cuda --training qlora` | NVIDIA CUDA 환경에서 NF4 4비트 추론·QLoRA |
| `--backend transformers --training lora` | CUDA가 있으면 CUDA, 없으면 CPU에서 비양자화 LoRA |
| `--no-activate` | 학습·평가를 수행하되 현재 웹에서 사용하는 활성 모델을 바꾸지 않음 |

Transformers의 `--device auto`는 설치된 PyTorch가 CUDA를 사용할 수 있으면 CUDA를 선택하고 그 외에는 CPU를 선택합니다. `--training auto`는 CUDA에서 QLoRA, CPU에서 LoRA를 선택합니다. 이 프로젝트의 CPU 경로는 QLoRA가 아닙니다. AMD/Intel GPU 및 MPS 가속은 현재 Transformers 경로에서 선택하지 않습니다.

CUDA 사용을 명시했는데 감지되지 않으면 오류로 중단합니다. 다음 순서로 CUDA PyTorch 빌드를 준비하세요.

1. `python -m venv .venv-torch`로 환경을 만듭니다.
2. Windows PowerShell: `.\.venv-torch\Scripts\Activate.ps1`; Linux: `source .venv-torch/bin/activate`. PowerShell 정책상 활성화할 수 없으면 `.\.venv-torch\Scripts\python.exe -m pip ...`처럼 인터프리터를 직접 사용합니다.
3. [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 OS와 드라이버에 맞는 CUDA 빌드의 설치 명령을 실행합니다.
4. 저장소에서 `python setup_local.py --backend transformers --device cuda --training qlora --serve`를 실행합니다. 나머지 패키지와 bitsandbytes는 설치 스크립트가 준비합니다.

CPU 전용으로 Linux·Windows의 큰 CUDA 패키지 다운로드를 피하려면 2단계까지 수행한 환경에서 `python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu`를 먼저 실행하고 `python setup_local.py --device cpu --serve`를 실행하세요. macOS에서는 PyPI의 macOS 휠을 사용합니다.

`--skip-install`은 해당 백엔드의 가상환경과 필요한 패키지(CUDA QLoRA면 bitsandbytes까지)가 이미 준비된 경우에만 사용하세요. 학습을 건너뛰는 옵션은 아닙니다. 설치 완료 뒤에는 `python run.py`로 바로 추론을 시작합니다.

## 구성

- MLX 모델: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`; 환경: `.venv-mlx/`; 의존성: `requirements-local.txt`.
- Transformers 모델: `Qwen/Qwen2.5-0.5B-Instruct`; 환경: `.venv-torch/`; 의존성: `requirements-torch.txt`.
- MLX는 `mlx-lm[train]==0.32.1`, Transformers는 `transformers==4.48.3`, `peft==0.14.0`, `accelerate==1.3.0`을 사용합니다. 실제 설치된 모든 하위 의존성은 각 실행의 `requirements-lock.txt`에 기록합니다.
- 모델 리비전: 다운로드 시 커밋 SHA를 조회하고 그 SHA로 다운로드합니다.
- 학습: 마지막 4개 레이어, 배치 1, 학습률 0.0001, 기본 80 iterations, prompt masking, gradient checkpointing. Transformers는 q/v projection에 rank 8 LoRA를 적용합니다. CUDA QLoRA는 bitsandbytes NF4와 double quantization을 사용하며, CPU는 FP32 LoRA입니다.
- 입력 최대 2048 토큰. 넘는 예제는 조용히 자르지 않고 중단합니다.
- 추론: greedy decoding, 최대 생성 96 토큰, 별도 Python 워커에 모델을 한 번 로드. 서버가 워커를 재사용하고 요청을 직렬화합니다.
- 추론에 추가 HTTP 서버나 포트가 필요하지 않습니다. Windows에서도 가능한 reader thread와 queue로 stdin/stdout JSON을 교환합니다. 모델 워커는 Hugging Face offline 모드, UTF-8로 실행합니다.
- MLX 추론 25초, Transformers 추론 120초 초과/동시 요청 대기 1초 초과/파싱 오류/잘못된 어휘는 규칙 추천으로 복구합니다. 브라우저는 서버가 알리는 제한에 맞춰 각각 35초/125초 대기합니다.

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
| `training.log` | 학습 loss, 검증 loss, 반복 횟수 |
| `before.json`, `after.json` | 학습 전후 유효 JSON 비율, hit@6, 추론 시간, 원시 출력 |
| `adapter/adapters.safetensors` | MLX가 저장한 실제 LoRA 가중치 |
| `adapter/adapter_model.safetensors`, `adapter/adapter_config.json` | PEFT가 저장한 실제 LoRA 가중치·설정 |
| `requirements-lock.txt` | 해당 실행의 설치 패키지 버전 |

`setup-status.json`이 `failed`이면 실행은 완료되지 않은 것입니다. 데이터 파일이 생성된 것과 모델이 학습된 것은 다릅니다. 학습 subprocess가 성공하고 어댑터 파일이 존재한 뒤에만 `training_completed: true`가 기록됩니다.

학습 후 모델이 유효한 추천을 하나도 생성하지 못하면 활성화하지 않습니다. 일부 유효 출력을 냈다고 기존 모델보다 개선됐다고 가정하지 않습니다. `before.json`과 `after.json`의 실패 사례·지연·적중을 직접 확인하세요. 현재 명령은 전후 비교에 test split을 사용하고 유효 출력 여부를 활성화 조건으로도 사용하므로, 엄밀한 최종 성능 보고에는 별도의 손대지 않은 holdout 세트가 필요합니다.

성공하면 `var/active-model.json`에 백엔드·양자화 방식·모델/어댑터 경로를 기록하고 원자적으로 바꿉니다. `--no-activate`를 지정하면 바꾸지 않습니다. 다음 `run.py` 실행은 해당 백엔드의 가상환경을 선택합니다. 백엔드 필드가 없는 과거 활성 모델 파일은 MLX로 해석합니다. `.env`와 과거 실행 디렉터리·가중치는 보존합니다.

## 다른 컴퓨터에서 재현

Git에는 코드와 합성 자료 생성 규칙만 포함합니다. `.venv-*`, `var/`, 사용자 DB, 모델 가중치, 어댑터는 포함하지 않습니다. 다른 사람이 clone한 뒤 설치 명령을 실행하면 자신의 OS에서 모델을 내려받고 새 어댑터를 학습합니다. 모델 리비전은 실행 시 조회하며 하위 패키지·하드웨어·훈련 난수의 영향도 있으므로 완전히 동일한 가중치나 추천 결과를 보장하지 않습니다.

MLX와 PEFT 어댑터는 서로 호환되는 파일이 아닙니다. 다른 OS로 옮길 때 `active-model.json`이나 `.venv-*`만 복사하지 말고 설치를 다시 진행하세요. macOS의 80회 MLX 학습과 별도의 Transformers CPU 80회 학습은 실제 Qwen 모델로 검증했습니다. CI는 Windows·Linux·Intel Mac·Apple Silicon에서 다운로드 없는 작은 Qwen 모델로 학습·저장·재로딩·워커 통신을 검사합니다. CUDA QLoRA는 별도 CUDA 기기에서 검증이 필요합니다.

## 수동 설정

`.env`의 기본값 `AAC_LOCAL_MODEL=auto`는 성공적으로 활성화한 모델을 읽습니다. 직접 경로를 지정할 수도 있습니다.

```dotenv
AAC_LOCAL_MODEL=/absolute/path/to/downloaded/model
AAC_LOCAL_ADAPTER=/absolute/path/to/run/adapter
# 수동 경로의 백엔드; auto는 OS로 선택
AAC_LOCAL_BACKEND=transformers
AAC_LOCAL_DEVICE=cpu
```

`AAC_LOCAL_MODEL=off`는 로컬 추론을 끕니다. 클라우드 옵션이 꺼져 있으면 규칙 추천을 사용합니다. 로컬 학습된 모델과 학습 전 베이스 모델을 비교하려면 `AAC_LOCAL_ADAPTER`를 비워 베이스 모델만 사용하세요. 로컬 모델에는 OpenAI API 키가 필요하지 않습니다.

## 참고

- [모델 카드](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit)
- [Qwen 원본 모델 카드](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [MLX LM LoRA 공식 문서](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX LM Python 추론 API](https://github.com/ml-explore/mlx-lm#python-api)
- [PEFT 양자화 학습](https://huggingface.co/docs/peft/en/developer_guides/quantization)
- [bitsandbytes 플랫폼 지원](https://huggingface.co/docs/bitsandbytes/en/installation)
