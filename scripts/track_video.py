# 영상 캐릭터 추적 + "내 캐릭터" 락온 (A안 후처리의 영상 버전)
# v2/v3 모델 + ByteTrack으로 추적하고, user_id가 보일 때 그 아래 character의 트랙 ID에
# 락온하여 이후 프레임에서 같은 ID를 따라간다. 라벨 없이 측정 가능한 추적 지표를 산출한다.
#
# 사용:
#   python scripts/track_video.py --video "<path>" [--weights <best.pt>] [--out out.mp4]
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_W = ROOT / "runs" / "df_yolo11s_1280_v2" / "weights" / "best.pt"


def pick_my_character(chars, ids):
    """가장 신뢰도 높은 user_id 아래에 정렬된 character를 내 캐릭터로 선정 (webapp 규칙과 동일)"""
    best = None
    for uid in ids:
        ucx = (uid["x1"] + uid["x2"]) / 2
        for ch in chars:
            ccx = (ch["x1"] + ch["x2"]) / 2
            dx = abs(ccx - ucx)
            dy = ch["y1"] - uid["y2"]
            if dy < -40 or dy > 150:
                continue
            if dx > (ch["x2"] - ch["x1"]):
                continue
            score = uid["conf"] - dx / 500 - abs(dy) / 500
            if best is None or score > best["score"]:
                best = {"score": score, "character": ch, "user_id": uid}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--out", default=None, help="주석 영상 출력 경로 (생략 시 미출력)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--relock-margin", type=float, default=0.05,
                    help="현재 락 트랙보다 이만큼 더 좋은 후보여야 도전자로 인정")
    # 07-25 스윕 기준 균형점(추적률 0.72 / ID스위치 88, 기존 276 대비 -68%). 영상별 튜닝 가능.
    ap.add_argument("--coast-grace", type=int, default=10,
                    help="락 트랙이 사라져도 이 프레임 수만큼은 같은 ID 복귀를 기다리며 재획득 보류")
    ap.add_argument("--challenge-frames", type=int, default=6,
                    help="다른 후보가 이 프레임 수만큼 연속 우세해야 락 교체 (진동 억제)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))

    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    locked_id = None        # 현재 내 캐릭터로 락온된 트랙 ID
    locked_score = None
    n = 0
    frames_my_located = 0   # 내 캐릭터(락 ID)가 화면에 잡힌 프레임 수
    frames_any_char = 0
    frames_any_uid = 0
    id_switches = 0
    coast_run = 0           # 락 ID가 연속으로 사라진 프레임 수
    max_coast = 0
    challenger_id = None    # 락 교체를 노리는 도전 트랙 (진동 억제용)
    challenger_count = 0

    results = model.track(
        source=args.video, imgsz=args.imgsz, conf=args.conf,
        persist=True, tracker=args.tracker, stream=True, verbose=False,
    )

    for r in results:
        n += 1
        chars, ids, by_tid = [], [], {}
        boxes = r.boxes
        tids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        for box, cls, c, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), tids):
            d = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                 "conf": c, "tid": int(tid) if tid is not None else None}
            if int(cls) == 0:
                chars.append(d)
                if d["tid"] is not None:
                    by_tid[d["tid"]] = d
            else:
                ids.append(d)

        if chars:
            frames_any_char += 1
        if ids:
            frames_any_uid += 1

        # user_id 근거로 내 캐릭터 후보 산출
        my = pick_my_character(chars, ids)
        cand_tid = my["character"]["tid"] if my else None
        cand_score = my["score"] if my else None

        locked_present = locked_id is not None and locked_id in by_tid
        if locked_present:
            # 락 유지(관성): 다른 후보가 challenge_frames 동안 '연속' 우세할 때만 교체.
            # 파티 장면에서 최고신뢰 user_id가 프레임마다 깜빡여 락이 튀던 문제를 억제한다.
            locked_score = by_tid[locked_id]["conf"]
            if (cand_tid is not None and cand_tid != locked_id and cand_score is not None
                    and cand_score > locked_score + args.relock_margin):
                if cand_tid == challenger_id:
                    challenger_count += 1
                else:
                    challenger_id, challenger_count = cand_tid, 1
                if challenger_count >= args.challenge_frames:
                    locked_id, locked_score = challenger_id, cand_score
                    challenger_id, challenger_count = None, 0
                    id_switches += 1
            else:
                challenger_id, challenger_count = None, 0
        else:
            # 락이 없거나 사라짐. 유예기간(coast_grace) 내엔 같은 ID 복귀를 기다리며 재획득 보류
            # (ByteTrack track_buffer로 잠깐 가려졌다 같은 ID로 돌아오는 경우 대비)
            if locked_id is None or coast_run >= args.coast_grace:
                if cand_tid is not None:
                    if locked_id is not None and cand_tid != locked_id:
                        id_switches += 1
                    locked_id, locked_score = cand_tid, cand_score
                    challenger_id, challenger_count = None, 0

        # 락 ID가 이번 프레임에 존재하는가
        my_box = by_tid.get(locked_id) if locked_id is not None else None
        if my_box is not None:
            frames_my_located += 1
            coast_run = 0
        else:
            coast_run += 1
            max_coast = max(max_coast, coast_run)

        if writer is not None:
            frame = r.plot()
            if my_box is not None:
                cv2.rectangle(frame, (int(my_box["x1"]), int(my_box["y1"])),
                              (int(my_box["x2"]), int(my_box["y2"])), (0, 0, 255), 3)
                cv2.putText(frame, f"MINE id={locked_id}", (int(my_box["x1"]), int(my_box["y1"]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"MINE coasting (id={locked_id})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(frame)

    if writer is not None:
        writer.release()

    metrics = {
        "video": Path(args.video).name,
        "weights": Path(args.weights).parent.parent.name,
        "frames": n,
        "my_located_rate": round(frames_my_located / n, 3) if n else 0,
        "any_char_rate": round(frames_any_char / n, 3) if n else 0,
        "any_uid_rate": round(frames_any_uid / n, 3) if n else 0,
        "id_switches": id_switches,
        "max_coast_frames": max_coast,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    out_json = ROOT / "outputs" / f"track_metrics_{metrics['weights']}_{Path(args.video).stem[:20]}.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
