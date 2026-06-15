# 색기반(HSV) 닉네임 추적 — YOLO 없이 user_id(노란 닉네임)만 색으로 잡아 프레임 간 ID 유지.
# 사용자가 보여준 라이브 캡처 코드(dnf.grab + inRange + dilate + findContours)를
#   ① 영상 파일 입력(cv2.VideoCapture)으로 바꾸고
#   ② 매 프레임 독립 탐지였던 것에 IoU 매칭 트래커를 붙여 "진짜 추적"(ID 연결)으로 확장한 버전.
#
# 색기반은 character(직업·코스튬마다 색이 달라 단일 범위로 못 잡음)에는 부적합하다.
# 캐릭터까지 제대로 추적하려면 track_video.py(v3 YOLO + ByteTrack)를 쓸 것. (CLAUDE.md A안 참조)
#
# 사용:
#   # 추적 + 주석 영상 출력
#   python scripts/track_nickname_color.py --video "datasets/df/videos/<file>" --out outputs/track_color.mp4
#   # HSV 범위 튜닝(마스크 미리보기, q로 종료)
#   python scripts/track_nickname_color.py --video "<file>" --tune
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 노란 닉네임 기본 HSV 범위(OpenCV: H 0-179, S/V 0-255). --hsv-lo/--hsv-hi로 튜닝.
DEFAULT_LO = (18, 80, 120)
DEFAULT_HI = (40, 255, 255)


def iou(a, b):
    """두 박스 [x1,y1,x2,y2]의 IoU"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class Track:
    __slots__ = ("id", "box", "age", "hits")

    def __init__(self, tid, box):
        self.id = tid
        self.box = box
        self.age = 0     # 마지막으로 매칭된 이후 경과 프레임
        self.hits = 1    # 누적 매칭 횟수(확정 판단용)


class IoUTracker:
    """경량 IoU 매칭 트래커(Kalman 없는 SORT-lite). 닉네임은 거의 정지 박스라 IoU만으로 충분."""

    def __init__(self, iou_thresh=0.3, max_age=15, min_hits=3):
        self.iou_thresh = iou_thresh
        self.max_age = max_age      # 이 프레임 수만큼 안 보이면 트랙 폐기(잠깐 가림 대비)
        self.min_hits = min_hits    # 이만큼 연속 잡혀야 확정(깜빡임 오탐 억제)
        self.tracks = []
        self._next_id = 1

    def update(self, dets):
        """dets: 박스 리스트 [x1,y1,x2,y2]. 반환: 확정 트랙 [(id, box), ...]"""
        # 그리디 IoU 매칭: (IoU 높은 순)으로 트랙↔탐지 1:1 연결
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                v = iou(t.box, d)
                if v >= self.iou_thresh:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].box = dets[di]
            self.tracks[ti].age = 0
            self.tracks[ti].hits += 1
            used_t.add(ti)
            used_d.add(di)

        # 매칭 안 된 탐지 → 새 트랙
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self._next_id, d))
                self._next_id += 1

        # 매칭 안 된 트랙 → 노화, 너무 오래되면 폐기
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.age += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return [(t.id, t.box) for t in self.tracks if t.hits >= self.min_hits and t.age == 0]


def detect_boxes(frame, lo, hi, kernel, min_area, min_ar):
    """HSV 색 마스크 → 팽창 → 윤곽 → 면적/가로세로비 필터 → 박스 리스트"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w / h < min_ar:   # 닉네임은 가로로 긴 형태 → 세로로 긴 노이즈 배제
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="주석 영상 출력 경로 (생략 시 미출력)")
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=list(DEFAULT_LO), metavar=("H", "S", "V"))
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=list(DEFAULT_HI), metavar=("H", "S", "V"))
    ap.add_argument("--min-area", type=int, default=400, help="이 면적 미만 덩어리는 노이즈로 버림")
    ap.add_argument("--min-ar", type=float, default=1.5, help="최소 가로/세로 비(닉네임은 가로로 긴 형태)")
    ap.add_argument("--dilate", type=int, default=6, help="팽창 커널 크기(글자 사이 틈 메우기)")
    ap.add_argument("--iou", type=float, default=0.3, help="트랙 매칭 IoU 임계값")
    ap.add_argument("--max-age", type=int, default=15, help="안 보여도 트랙 유지할 프레임 수")
    ap.add_argument("--min-hits", type=int, default=3, help="확정까지 필요한 연속 매칭 수")
    ap.add_argument("--tune", action="store_true", help="마스크 미리보기로 HSV 튜닝(저장 안 함, q 종료)")
    ap.add_argument("--start", type=int, default=0, help="이 프레임부터 처리(구간 추출용)")
    ap.add_argument("--frames", type=int, default=0, help="처리할 최대 프레임 수(0=끝까지)")
    args = ap.parse_args()

    lo = np.array(args.hsv_lo, dtype=np.uint8)
    hi = np.array(args.hsv_hi, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.dilate, args.dilate))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))

    # 튜닝 모드: 마스크와 박스를 화면에 띄워 HSV 범위를 눈으로 맞춤
    if args.tune:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, mask = detect_boxes(frame, lo, hi, kernel, args.min_area, args.min_ar)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    writer = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    tracker = IoUTracker(iou_thresh=args.iou, max_age=args.max_age, min_hits=args.min_hits)
    n = 0
    frames_with_det = 0
    seen_ids = set()

    while True:
        if args.frames and n >= args.frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        boxes, _ = detect_boxes(frame, lo, hi, kernel, args.min_area, args.min_ar)
        if boxes:
            frames_with_det += 1
        confirmed = tracker.update(boxes)
        for tid, box in confirmed:
            seen_ids.add(tid)
            if writer is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"id={tid}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    metrics = {
        "video": Path(args.video).name,
        "method": "hsv_color + iou_tracker",
        "hsv_lo": args.hsv_lo,
        "hsv_hi": args.hsv_hi,
        "frames": n,
        "det_rate": round(frames_with_det / n, 3) if n else 0,  # 닉네임이 1개 이상 잡힌 프레임 비율
        "unique_ids": len(seen_ids),                            # 영상 전체에서 부여된 고유 트랙 수
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    out_json = ROOT / "outputs" / f"track_color_{Path(args.video).stem[:20]}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
