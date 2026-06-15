# 웹 애플리케이션 고도화 계획 — 던파 캐릭터 탐지 & 추적

## Context (왜 하는가)

현재 산출물은 셋으로 흩어져 있다: Flask 스크린샷 앱(`webapp/app.py` :5000), Flask 영상 추적 앱
(`webapp/video_app.py` :5001), Streamlit 포트폴리오(`portfolio_app.py`). ML 코어(탐지·ByteTrack·hysteresis
락·지표 계산)는 검증됐고(mAP50 0.931, ID스위치 276→88), 부족한 건 **"하나의 완성된 제품"으로서의 구조와
프레젠테이션**이다.

**목표**: 채용·동료에게 보여줄 수준의 모던 웹앱. 멋진 쇼케이스(스토리텔링) + 실제로 이미지/영상을 업로드해
탐지·추적이 돌아가는 라이브 데모를, **현업 표준 스택(React/Next.js + FastAPI)**으로, **로컬 구동**으로 만든다.

**결정 사항** (사용자 확정): 목적=쇼케이스+동작 / 스택=모던 SPA(Next.js) / 배포=로컬 데모만.

---

## 아키텍처

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  Frontend (Next.js, :3000)  │  HTTP  │  Backend (FastAPI, :8000)    │
│  - App Router + TypeScript  │ ─────► │  - YOLO v3 모델 1회 로드      │
│  - Tailwind + shadcn/ui     │  proxy │  - /api/detect  (이미지)      │
│  - 랜딩/케이스스터디/라이브  │ ◄───── │  - /api/track/* (영상 스트림) │
│  - Canvas 박스 오버레이      │ stream │  - /api/demos   (사전 산출물) │
└─────────────────────────────┘        │  - core/ 공유 추론 모듈       │
                                        └──────────────────────────────┘
```

- 기존 두 Flask 앱 + Streamlit 포트폴리오 → **단일 FastAPI 백엔드 + 단일 Next.js 프론트**로 통합.
- 기존 `webapp/`, `portfolio_app.py`는 **삭제하지 않고 보존**(레거시 참고용). 신규는 `web/` 하위에 구성.
- 로컬 전용: Next dev가 `/api/*`를 FastAPI(:8000)로 프록시. 단일 `start_web.bat`로 둘 다 기동.

### 고정 제약 준수 (CLAUDE.md)
- `imgsz=1280` 그대로 (낮추지 않음).
- 모델 경로 `runs/df_yolo11s_1280_v3/weights/best.pt` (환경변수로 파라미터화하되 기본값 v3).
- GPU 8GB(2070 SUPER)·전원 이슈 → **추론 동시성 1로 제한**(영상 추적 작업 큐, 동시 1건). 추론은 부하 낮아 안전.
- 기존 파이프라인 스크립트(`scripts/`)는 건드리지 않음.

---

## 단계별 계획

### Phase 0 — 공유 추론 코어 추출 (리팩터링, DRY)
현재 `pick_my_character()`와 추론 로직이 `scripts/track_video.py`와 `webapp/app.py`에 **중복**돼 있다.
- `web/backend/core/inference.py` 생성: 모델 로드, `detect_image()`, `pick_my_character()` 단일 정의.
- `web/backend/core/tracking.py` 생성: hysteresis 락 상태머신 + 지표 계산(`video_app.py`에서 추출).
- 기존 `scripts/track_video.py`도 이 코어를 import하도록 정리(선택 — 호환 깨지지 않는 범위에서).

### Phase 1 — FastAPI 백엔드 통합
`web/backend/main.py`:
- lifespan에서 `YOLO(WEIGHTS)` 1회 로드.
- `POST /api/detect` — 이미지 업로드 → `{image:{w,h}, characters[], user_ids[], my_character}` (app.py 로직 이식).
- `POST /api/track/upload` → token / `GET /api/track/stream/{token}` MJPEG / `GET /api/track/metrics/{token}`
  / `GET /api/track/download/{token}` (video_app.py 로직 이식, in-memory 잡 + asyncio 동시성 락).
- `GET /api/demos` — `outputs/*.mp4` + `*_metrics_*.json` 메타 반환(사전 계산 데모용).
- `GET /api/version` — 모델 버전·mAP50 등 메타.
- 정적 서빙: `outputs/`, `reports/*_assets/` 자산을 `/static/`으로 노출.
- 에러 핸들링(HTTP 상태코드·검증), CORS, 구조적 로깅 추가.
- 의존성: `fastapi`, `uvicorn[standard]`, `python-multipart` → `requirements.txt` 추가.

### Phase 2 — Next.js 프론트 스캐폴드 + 디자인 시스템
`web/frontend/` (Next.js App Router, TypeScript):
- Tailwind + shadcn/ui + framer-motion. 다크/네온 테마(기존 portfolio 색: cyan #00e5a8, blue #3b82f6 계승).
- 공통 레이아웃: 상단 네비(로고·섹션 앵커·GitHub), 푸터, 반응형 그리드.
- `next.config` rewrites로 `/api/*` → `http://127.0.0.1:8000` 프록시.

### Phase 3 — 랜딩 / 케이스 스터디 (쇼케이스, portfolio_app.py 대체)
- **Hero**: 헤드라인 + 핵심 지표 카드(mAP50 0.931, ID스위치 −68%, 학습데이터 규모) + 기술 배지.
- **How it works**: 5단계 파이프라인 다이어그램, A안(후처리 내 캐릭터 선정) 설명.
- **Results**: 버전 히스토리 표(v1→v3) + `reports/report_v2_v3_assets/` 비교 이미지 + recharts 지표 차트.
- **Demo gallery**: `outputs/track_v3_hysteresis_0725.mp4` vs baseline 나란히 비교, 사전계산 지표 표시.
- **Hardware resilience**: 전원 이슈→청크 학습 스토리(`report_hardware_assets/` 다이어그램). 차별화 포인트.

### Phase 4 — 라이브 탐지 페이지 (스크린샷)
- 드래그앤드롭 업로드 → `/api/detect` → **Canvas 박스 오버레이**(character 빨강 / user_id 시안 / 내캐릭 초록).
- conf 슬라이더, 결과 표, 샘플 이미지 버튼(`labeled/`에서 몇 장 동봉해 "예시로 시도").

### Phase 5 — 라이브 추적 페이지 (영상)
- 영상 업로드 → 진행 상태 → MJPEG 스트림 표시 + **실시간 지표 카드**(탐지율·추적률·ID스위치, 1s 폴링 또는 SSE).
- 완료 후 주석 mp4 다운로드. 짧은 샘플 클립 동봉(긴 영상은 8GB·전원 고려해 권장 길이 안내).

### Phase 6 — 마감 (현업 디테일)
- 로딩 스켈레톤, 에러 토스트, 빈 상태, 모바일 반응형, 접근성(alt·키보드).
- `web/README.md`: 셋업·실행법. `start_web.bat`(백엔드+프론트 동시 기동).
- 선택: Dockerfile 초안(로컬 전용이나 "공유 가능 구조"로 가치↑) — 우선순위 낮음.

---

## 신규/변경 파일 (핵심)

```
web/
  backend/
    main.py                FastAPI 진입점
    core/inference.py      detect_image, pick_my_character (단일 정의)
    core/tracking.py       hysteresis 락 + 지표
    requirements 추가      fastapi, uvicorn, python-multipart
  frontend/
    app/                   App Router 페이지(랜딩/탐지/추적/케이스스터디)
    components/            UI 컴포넌트(shadcn 기반)
    next.config.*          /api 프록시
  README.md
start_web.bat              백엔드+프론트 동시 기동
```
- 보존(미변경): `webapp/`, `portfolio_app.py`, `scripts/`, 모델·데이터셋.

## 재사용 자산
- 추론·선정 로직: `webapp/app.py` `pick_my_character()`, `scripts/track_video.py`.
- 추적·스트리밍·지표: `webapp/video_app.py`(MJPEG, JOBS, hysteresis 파라미터 COAST_GRACE=10 등).
- 트래커 설정: `scripts/bytetrack_df.yaml`.
- 데모 자산: `outputs/*.mp4`·`*_metrics_*.json`, `reports/report_v2_v3_assets/`, `reports/report_hardware_assets/`.
- 디자인 토큰·문구: `portfolio_app.py`의 색상·섹션 구성.

---

## 검증 (E2E)
1. `start_web.bat` → 백엔드 :8000, 프론트 :3000 기동 확인.
2. 라이브 탐지: `labeled/`의 스크린샷 업로드 → Canvas에 박스·내 캐릭터(초록) 표시, 결과표 일치.
3. 라이브 추적: `datasets/df/videos/`의 짧은 클립 업로드 → 스트림 재생 + 지표(추적률·ID스위치) 갱신 → mp4 다운로드.
4. 케이스 스터디: 데모 갤러리·비교 이미지·차트 로드 확인(자산 없으면 graceful 메시지).
5. 동시성: 영상 작업 2건 연속 업로드 시 큐로 직렬 처리(OOM 없음) 확인.
6. 회귀: 기존 `webapp/start_webapp.bat` 등 레거시 경로 무손상 확인.

## 권장 진행 순서 / 우선순위
Phase 0→1(백엔드 통합)을 먼저 끝내 **동작하는 API**를 확보 → Phase 2~3(보이는 쇼케이스) → Phase 4~5(라이브)
→ Phase 6(마감). 각 Phase 끝에서 실행 가능한 상태를 유지(증분 검증).

## 확정 사항
- 스택: 모던 SPA(Next.js) + FastAPI / 로컬 데모만 / 쇼케이스+동작 겸용.
- **Node.js LTS 설치 후 진행**(사용자 확정). 첫 시스템 변경 작업 = Node.js 설치.
- UI 언어: 기본 한국어(필요시 i18n).
