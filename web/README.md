# DNF Vision — 웹 애플리케이션

던파 캐릭터 탐지·추적 프로젝트의 통합 웹앱. **FastAPI 백엔드 + Next.js 프론트엔드**로,
프로젝트 쇼케이스(케이스 스터디)와 실제 라이브 추론(스크린샷 탐지 · 영상 추적)을 한곳에서 제공한다.

> 기존 `webapp/`(Flask)·`portfolio_app.py`(Streamlit)를 대체하는 신규 구현. 레거시는 그대로 보존된다.

## 구조

```
web/
  backend/                 FastAPI (:8000)
    main.py                API 진입점 (detect / track / demos / samples)
    core/
      config.py            경로·모델·하이퍼파라미터 단일 출처
      inference.py         detect_image · pick_my_character (단일 정의)
      tracking.py          hysteresis 락 + 지표 (프레임 제너레이터)
    _work/                 업로드/산출 임시 파일 (gitignore 대상)
  frontend/                Next.js 16 · React 19 · Tailwind v4 (:3000)
    src/app/               page(랜딩) · detect · track
    src/components/        nav · footer · ui · version-chart · demo-gallery
    src/lib/api.ts         백엔드 호출 헬퍼 (NEXT_PUBLIC_API_BASE)
```

## 실행

### 한 번에 (권장)
프로젝트 루트의 **`start_web.bat`** 더블클릭 → 백엔드·프론트가 각각 새 창에서 뜨고 브라우저가 열린다.

### 수동
```powershell
# 백엔드
cd web\backend
..\..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# 프론트엔드 (별도 터미널, Node.js 필요)
cd web\frontend
npm install   # 최초 1회
npm run dev
```
브라우저에서 http://localhost:3000

## API 요약 (백엔드 :8000)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/version` | 모델 버전·mAP50·락 파라미터 |
| POST | `/api/detect` | 이미지 업로드 → 탐지 결과(JSON) |
| GET | `/api/detect_sample?name=` | 샘플 스크린샷 서버측 탐지 |
| POST | `/api/track/upload` | 영상 업로드 → token |
| GET | `/api/track/stream/{token}?pace=` | MJPEG 추적 스트림 (서버측 추적 시작) |
| GET | `/api/track/metrics/{token}` | 추적 지표 폴링 |
| GET | `/api/track/download/{token}` | 주석 mp4 다운로드 |
| GET | `/api/demos` | 사전 계산 데모 영상·지표 |
| GET | `/api/samples/images` | 라이브 탐지 샘플 목록 |

정적 자산: `/static/outputs/*` (데모 mp4), `/static/reports/*` (보고서 이미지), `/static/labeled/*` (샘플).

## 설계 메모

- **모델 로드 1회**: FastAPI lifespan에서 v3 가중치 워밍업. `core.inference.get_model()` 싱글톤.
- **GPU 동시성 1**: `INFER_LOCK`으로 추론을 동시 1건만 수행(8GB VRAM·전원 이슈 대응).
- **탐지 박스 렌더**: 프론트는 canvas 대신 **이미지 위 퍼센트 좌표 div 오버레이** — cross-origin/taint 문제 없이 반응형.
- **추적 코어 통합**: `scripts/track_video.py`·`webapp/video_app.py`에 중복됐던 락 상태머신을 `core/tracking.py` 제너레이터로 단일화.
- **API 베이스**: 프론트 기본값 `http://127.0.0.1:8000`. 바꾸려면 `web/frontend/.env.local`에 `NEXT_PUBLIC_API_BASE=` 지정.

## 무료 배포 — Cloudflare Tunnel

이 앱의 백엔드는 Python+PyTorch+YOLO라 Cloudflare 서버리스(Pages/Workers)에 못 올린다.
대신 **로컬에서 실행하고 `cloudflared`로 공개 URL만 뚫는다**. 프론트가 `/api`·`/static`을 백엔드로
프록시(`next.config.ts` rewrites)하므로 **프론트(:3000) 하나만 터널하면 전체가 공개**된다(CORS 불필요).

### 실행
1. `cloudflared` 설치(최초 1회): `winget install Cloudflare.cloudflared`
2. 루트의 **`start_tunnel.bat`** 더블클릭 → 백엔드·프론트·터널 3개 창이 뜬다.
3. "DNF Vision Tunnel" 창에 출력되는 `https://....trycloudflare.com` 주소를 공유.

세 창이 열려 있는 동안(=PC가 켜져 있는 동안)만 접속 가능하다.

### 안정적인 고정 주소 (선택)
quick tunnel 주소는 재시작마다 바뀐다. 고정 주소가 필요하면 **무료 Cloudflare 계정 + 도메인**으로 named tunnel:
```
cloudflared login
cloudflared tunnel create dnf-vision
cloudflared tunnel route dns dnf-vision dnf.<your-domain>
cloudflared tunnel run --url http://localhost:3000 dnf-vision
```

### 더 견고한 배포(선택)
공개 데모를 오래 띄울 거면 `next dev` 대신 프로덕션 빌드를 권장:
```
npm --prefix web\frontend run build
npm --prefix web\frontend start
```

## 주의

- `imgsz=1280` 고정(닉네임 탐지 한계). 모델 경로는 `DF_WEIGHTS` 환경변수로 덮어쓰기 가능(기본 v3).
- 영상 추적은 GPU 부하가 있으므로 데모는 **10~30초 클립** 권장.
- 공개 URL은 누구나 접속 가능하다 — 추론은 내 GPU를 쓰므로, 원치 않으면 터널 창을 닫아 즉시 차단한다.
