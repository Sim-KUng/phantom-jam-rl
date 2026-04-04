# 🚗 Phantom Jam (유령 정체) 해소 강화학습 프로젝트

본 레포지토리는 2차선 고속도로 환경에서 발생하는 '유령 정체(Phantom Jam)' 현상을 시뮬레이션하고, 소수(5%)의 자율주행 강화학습 에이전트(PPO)를 투입하여 교통 흐름을 최적화하는 프로젝트입니다.

## 📁 프로젝트 폴더 구조

최근 코드 모듈화 및 데이터 관리를 위해 폴더 구조가 개편되었습니다.

```text
phantom-jam-rl/
├── src/                      # 소스 코드 폴더
│   ├── custom_highway.py     # [1단계] 베이스라인 시뮬레이션 & 정체 유발
│   ├── visualize_log.py      # [2단계] CSV 기반 시공간 속도 저하 그래프 생성
│   ├── week4_env_wrapper.py  # [3단계] 다차선 State/Action 래퍼 테스트
│   └── test_env.py           # 기본 환경 테스트용
├── logs/                     # [자동생성] 시뮬레이션 CSV 로그 저장 폴더
├── results/                  # [자동생성] 시각화 이미지(png) 저장 폴더
├── venv/                     # 파이썬 가상환경 폴더
├── requirements.txt          # 패키지 의존성 파일
└── README.md                 # 현재 파일
```

---

## 🚀 파일 실행 순서 가이드

**⚠️ 주의:** 모든 파이썬 실행 명령어는 반드시 최상위 폴더인 `phantom-jam-rl` 경로에서 실행해야 합니다. (그래야 `logs/` 및 `results/` 폴더가 정상적으로 자동 생성됩니다.)

먼저 가상환경을 켜주세요.
* Windows: `.\venv\Scripts\activate`
* Mac/Linux: `source venv/bin/activate`

### Step 1. 베이스라인 정체 시뮬레이션 실행
100% 인간 운전자(IDM) 차량만 있는 환경에서, 느린 차량(빌런) 트리거를 통해 유령 정체를 발생시킵니다.
```bash
python src/custom_highway.py
```
* **결과:** Pygame 렌더링 창이 뜨며 시뮬레이션이 진행됩니다. 종료 후 `logs/baseline_log.csv` 파일이 자동 생성됩니다.

### Step 2. 정체 데이터 시각화 (그래프 추출)
Step 1에서 수집된 로그 데이터를 바탕으로, 시간 흐름에 따른 평균 속도 저하(Shockwave) 그래프를 그립니다.
```bash
python src/visualize_log.py
```
* **결과:** `results/speed_degradation.png` 파일이 생성되며, 로컬 화면에 그래프 창이 팝업됩니다.

### Step 3. 다차선 환경 Wrapper 통신 테스트 (4주 차)
강화학습 에이전트가 사용할 `Kinematic Observation (State)`과 `Discrete Meta Action (Action)`이 OpenAI Gym 규격에 맞게 통신하는지 검증합니다.
```bash
python src/week4_env_wrapper.py
```
* **결과:** 터미널에 State 형태 `(15, 5)` 및 Action Space가 출력되며, 디버깅용 랜덤 액션 루프가 돌아간 후 `logs/phantom_jam_log_week4.csv` 파일이 저장됩니다.

---

## 🛠️ 개발 환경 설정 (초기 세팅용)

이 레포지토리를 처음 클론 받은 팀원은 아래 명령어를 통해 환경을 동기화해 주세요.

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화 (Windows 기준)
.\venv\Scripts\activate

# 3. 필수 패키지 설치
pip install -r requirements.txt
```