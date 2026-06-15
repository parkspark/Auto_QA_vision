# 영상에서 학습용 프레임 추출 + v2 pseudo-label 생성
# - fps_sample 간격으로 프레임 추출
# - v2 모델로 추론하여 character/user_id pseudo-label(YOLO txt) 생성
# - 메뉴/정산 화면(탐지 0) 자동 제외
# - "닉네임은 있으나 캐릭터 미탐지"인 어려운 프레임은 review_pool에 별도 기록(사람 검수용)
#
# 07-25(holdout)는 넣지 말 것. train 비오염 유지.
#
# 출력: datasets/df_video/images/*.jpg, datasets/df_video/labels/*.txt
#       datasets/df_video/review_pool.csv (사람 검수 후보)
import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_W = ROOT / "runs" / "df_yolo11s_1280_v2" / "weights" / "best.pt"
OUT = ROOT / "datasets" / "df_video"
VID_DIR = ROOT / "datasets" / "df" / "videos"

PSEUDO_CONF = 0.50   # 학습 라벨로 채택할 최소 신뢰도 (보수적)
GATE_CONF = 0.35     # 게임플레이 프레임 판정용(메뉴 제외)
HOLDOUT = "dnfvideo 2025-07-25 00-03-38-255.avi"  # 평가 전용, 추출 금지


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps-sample", type=float, default=0.5, help="초당 추출 프레임 수")
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--max-per-video", type=int, default=800)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    vids = [p for p in sorted(VID_DIR.iterdir())
            if p.suffix.lower() in (".mp4", ".avi") and p.name != HOLDOUT]
    print(f"train videos: {[v.name for v in vids]}")

    kept = 0
    review_rows = []
    for vp in vids:
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / args.fps_sample)))
        tag = vp.stem.replace(" ", "_")
        vid_kept = 0
        for i in range(0, n, step):
            if vid_kept >= args.max_per_video:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            H, Wd = frame.shape[:2]
            r = model.predict(frame, imgsz=1280, conf=GATE_CONF, verbose=False)[0]
            lines = []
            n_char = n_uid = 0
            for box, cls, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if c < PSEUDO_CONF:
                    continue
                k = int(cls)
                if k == 0:
                    n_char += 1
                else:
                    n_uid += 1
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2 / Wd, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / Wd, (y2 - y1) / H
                lines.append(f"{k} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if n_char == 0:
                # 메뉴(탐지 전무) vs 어려운 프레임(닉네임만 존재) 구분
                if n_uid > 0:
                    name = f"vid_{tag}_{i:06d}"
                    fpath = OUT / "images" / f"{name}.jpg"
                    cv2.imwrite(str(fpath), frame)
                    review_rows.append([name, vp.name, i, n_uid])
                continue  # 캐릭터 pseudo-label이 없으면 학습셋에서 제외

            name = f"vid_{tag}_{i:06d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), frame)
            (OUT / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            kept += 1
            vid_kept += 1
        cap.release()
        print(f"  {vp.name}: kept {vid_kept} labeled frames")

    if review_rows:
        with (OUT / "review_pool.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "video", "frame_idx", "n_uid"])
            w.writerows(review_rows)
    print(f"\nTOTAL labeled frames: {kept}")
    print(f"review_pool (uid but no char): {len(review_rows)} frames -> datasets/df_video/review_pool.csv")


if __name__ == "__main__":
    main()
