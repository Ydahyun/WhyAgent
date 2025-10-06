# 🧠 WhyAgent - "AAPL이 왜 이렇게 예측돼?"

AI 기반 주가 예측 + 최신 뉴스 기반 설명을 제공하는 **FastAPI 기반 LLM Agent**입니다.  
ChatGPT 스타일로 “왜 이렇게 예측됐는지”를 자연어로 설명합니다.

---

## 🔧 주요 기능

- **시세 수집**  
  - `yfinance` 기반 실시간 주가 수집 (Stooq fallback 지원)
  - `tickers.yml`을 통한 다중 티커 설정 및 자동 수집
  - (`Airflow`로 자동화 예정)
- **데이터 전처리 및 피처 엔지니어링**
  - 수익률, 거래량 변화율, 주중/주말 등 파생 피처 생성
- **모델 학습**
  - `XGBoost` 기반 회귀 모델 학습
  - `MLflow`로 모델 성능 및 아티팩트 관리
- **예측 및 설명 API**
  - `/predict`: 다음날 종가 예측
  - `/explain`: 예측 결과에 대한 자연어 설명 생성
  - `/api/chat`: 뉴스 기반 ReAct 스타일 문답형 설명
- **웹 UI 제공**
  - `/web/` 접속 시 간단한 주가 예측 & 설명 웹 인터페이스 제공

---

## 📁 프로젝트 구조

WhyAgent/  
├── app/  
│   └── Services/  
│       ├── main.py              # FastAPI 엔트리포인트 (라우트)  
│       ├── chat.py              # /api/chat 핸들러  
│       ├── explain.py           # 뉴스 수집 + LLM 프롬프트/호출  
│       └── mlflow_loader.py     # 모델 로딩/예측 헬퍼  
│  
├── web/  
│   └── index.html               # 간단한 채팅 UI (/web/)  
│  
├── ml/  
│   ├── fetch_prices.py          # yfinance(+Stooq) 수집  
│   ├── features.py              # 피처 생성/표준화  
│   └── train.py                 # XGBoost 학습 + MLflow 로깅  
│  
├── utils/  
│   └── io_utils.py              # data 경로 등 I/O 유틸  
│  
├── configs/  
│   └── tickers.yml              # 티커/주기 설정  
│  
├── data/  
│   └── prices/                  # 저장된 Parquet 파일  
│  
├── .env                         # 환경 변수  
├── requirements.txt  
└── README.md  
  


## 🚧 아키텍쳐  
  
              ▼  
          fetch_prices.py  
              │  
    ┌─────────┴──────────┐  
    ▼                    ▼  
Yahoo (yfinance)       Stooq  
    ▼                    ▼  
  [수집 실패 시 fallback]  
    ▼  
.parquet 저장 (data/prices/*.parquet)  
    ▼  
make_features.py → features 생성  
    ▼  
train.py → XGBoost 모델 학습 + MLflow 로깅  
    ▼  
MLflow에서 모델 로딩  
    ▼  
FastAPI 서버 (main.py)  
    ├── /predict   ← API 호출 (JSON)  
    ├── /api/chat  ← 자연어 입력 (LLM explain)  
    └── /web/      ← UI 인터페이스  
  
