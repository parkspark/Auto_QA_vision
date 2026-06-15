"""던파 캐릭터 탐지·추적 — 통합 FastAPI 백엔드.

기존 webapp/app.py(:5000, 스크린샷)와 webapp/video_app.py(:5001, 영상)를 단일 서비스로 통합하고,
프론트(Next.js)가 쓸 데모/버전 메타 API와 정적 자산 서빙을 추가한다.

실행: web/backend 에서  uvicorn main:app --port 8000
GPU(2070 SUPER 8GB)·전원 이슈 대응으로 추론은 INFER_LOCK으로 동시 1건만 수행한다.
"""
from __future__ import annotations

import io
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from core import config
from core.inference import detect_image, get_model
from core.tracking import track_video

# GPU 동시성 1 제한 — 8GB VRAM에서 추론 충돌(OOM) 방지
INFER_LOCK = threading.Lock()

# 영상 추적 잡 상태 (in-memory; 로컬 데모 전용)
JOBS: dict[str, dict] = {}

config.WORK_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 워밍업 (첫 요청 지연 제거)
    get_model()
    yield


app = FastAPI(title="DNF Detector API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 자산: 데모 영상 / 보고서 이미지 / 샘플 스크린샷
app.mount("/static/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")
app.mount("/static/reports", StaticFiles(directory=str(config.REPORTS_DIR)), name="reports")
if config.LABELED_DIR.exists():
    app.mount("/static/labeled", StaticFiles(directory=str(config.LABELED_DIR)), name="labeled")


@app.get("/api/version")
def version():
    return {
        "model_version": config.MODEL_VERSION,
        "map50": config.MODEL_MAP50,
        "imgsz": config.IMGSZ,
        "classes": {0: "character", 1: "user_id"},
        "lock": {
            "coast_grace": config.COAST_GRACE,
            "challenge_frames": config.CHALLENGE_FRAMES,
            "relock_margin": config.RELOCK_MARGIN,
        },
    }


# ---------------------------------------------------------------- 스크린샷 탐지
@app.post("/api/detect")
async def detect(image: UploadFile = File(...), conf: float = Form(0.5)):
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "빈 이미지입니다")
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 열 수 없습니다")
    with INFER_LOCK:
        return detect_image(im, conf)


# ---------------------------------------------------------------- 영상 추적
@app.post("/api/track/upload")
async def track_upload(video: UploadFile = File(...)):
    raw = await video.read()
    if not raw:
        raise HTTPException(400, "빈 영상입니다")
    token = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    src = config.WORK_DIR / f"{token}_in{suffix}"
    src.write_bytes(raw)
    JOBS[token] = {"src": str(src), "status": "uploaded", "metrics": None, "out": None}
    return {"token": token}


def _track_mjpeg(token: str, pace: bool):
    """추적하며 MJPEG 프레임을 흘리고, 동시에 mp4 기록 + 지표 갱신."""
    job = JOBS[token]
    out_path = config.WORK_DIR / f"{token}_out.mp4"
    writer = None
    job["status"] = "running"
    with INFER_LOCK:
        try:
            t_prev = time.time()
            for step in track_video(job["src"], draw=True):
                frame = step["frame"]
                job["metrics"] = step["metrics"]
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), step["fps"], (w, h)
                    )
                writer.write(frame)

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

                if pace:
                    interval = 1.0 / (step["fps"] or 30.0)
                    dt = time.time() - t_prev
                    if dt < interval:
                        time.sleep(interval - dt)
                    t_prev = time.time()
        finally:
            if writer is not None:
                writer.release()
            job["out"] = str(out_path)
            job["status"] = "done"


@app.get("/api/track/stream/{token}")
def track_stream(token: str, pace: int = 1):
    if token not in JOBS:
        raise HTTPException(404, "unknown token")
    return StreamingResponse(
        _track_mjpeg(token, pace != 0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/track/metrics/{token}")
def track_metrics(token: str):
    j = JOBS.get(token, {})
    return {"status": j.get("status", "unknown"), "metrics": j.get("metrics")}


@app.get("/api/track/download/{token}")
def track_download(token: str):
    j = JOBS.get(token)
    if not j or not j.get("out"):
        raise HTTPException(404, "not ready")
    return FileResponse(j["out"], filename="tracked.mp4", media_type="video/mp4")


# ---------------------------------------------------------------- 데모 / 샘플
@app.get("/api/demos")
def demos():
    """사전 계산된 추적 영상 + 지표 메타 (케이스 스터디 갤러리용)."""
    videos = []
    for mp4 in sorted(config.OUTPUTS_DIR.glob("*.mp4")):
        videos.append({
            "name": mp4.name,
            "url": f"/static/outputs/{mp4.name}",
            "size_mb": round(mp4.stat().st_size / 1e6, 1),
        })
    metrics = []
    for jf in sorted(config.OUTPUTS_DIR.glob("track_metrics_*.json")):
        try:
            metrics.append({"file": jf.name, **json.loads(jf.read_text(encoding="utf-8"))})
        except Exception:
            pass
    return {"videos": videos, "metrics": metrics}


@app.get("/api/samples/images")
def sample_images(limit: int = 8):
    """라이브 탐지 '예시로 시도'용 샘플 스크린샷 파일명 목록."""
    if not config.LABELED_DIR.exists():
        return {"images": []}
    names = [p.name for p in sorted(config.LABELED_DIR.glob("*.jpg"))[:limit]]
    return {"images": [{"name": n, "url": f"/static/labeled/{n}"} for n in names]}


@app.get("/api/detect_sample")
def detect_sample(name: str, conf: float = 0.5):
    """샘플 스크린샷을 서버에서 직접 열어 탐지 (브라우저 cross-origin blob 회피)."""
    safe = Path(name).name  # 경로 탈출 방지
    path = config.LABELED_DIR / safe
    if not path.exists():
        raise HTTPException(404, "샘플을 찾을 수 없습니다")
    with INFER_LOCK:
        return detect_image(Image.open(path).convert("RGB"), conf)


@app.get("/api/health")
def health():
    return {"ok": True}
