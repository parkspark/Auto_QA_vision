"""공유 설정 — 경로·모델·하이퍼파라미터의 단일 출처.

기존 webapp/app.py, webapp/video_app.py, scripts/track_video.py에 흩어져 있던
상수를 한곳으로 모은다. 모델 경로는 환경변수 DF_WEIGHTS로 덮어쓸 수 있으나 기본은 v3.
"""
import os
from pathlib import Path

# web/backend/core/config.py → parents[3] = 프로젝트 루트
ROOT = Path(__file__).resolve().parents[3]

# 현역 모델 (CLAUDE.md: v3 = mAP50 0.931). DF_WEIGHTS로 덮어쓰기 가능.
WEIGHTS = Path(os.environ.get("DF_WEIGHTS", ROOT / "runs" / "df_yolo11s_1280_v3" / "weights" / "best.pt"))
TRACKER = ROOT / "scripts" / "bytetrack_df.yaml"

# 데모/자산 디렉터리 (사전 계산된 추적 영상·보고서 이미지)
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
LABELED_DIR = ROOT / "labeled"
VIDEOS_DIR = ROOT / "datasets" / "df" / "videos"

# 업로드/산출 임시 디렉터리
WORK_DIR = ROOT / "web" / "backend" / "_work"

# 고정 결정 (CLAUDE.md): user_id(~70×24px)가 640에서 탐지 한계 → 1280 유지
IMGSZ = 1280

# 모델 메타 (UI 표시용)
MODEL_VERSION = "df_yolo11s_1280_v3"
MODEL_MAP50 = 0.931

# hysteresis 락 기본값 (track_video.py 07-25 스윕 균형점: 추적률 0.72 / ID스위치 88)
COAST_GRACE = 10        # 락 트랙이 사라져도 같은 ID 복귀를 기다리는 프레임 수
CHALLENGE_FRAMES = 6    # 다른 후보가 연속 우세해야 락 교체 (진동 억제)
RELOCK_MARGIN = 0.05    # 락 트랙보다 이만큼 우세해야 도전자로 인정
