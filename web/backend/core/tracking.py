"""영상 추적 + '내 캐릭터' hysteresis 락온.

scripts/track_video.py와 webapp/video_app.py에 중복돼 있던 락 상태머신을
프레임 단위 제너레이터로 통합한다. mp4 기록·MJPEG 인코딩은 API 계층이 담당한다.
"""
from __future__ import annotations

from typing import Iterator

import cv2

from .config import CHALLENGE_FRAMES, COAST_GRACE, IMGSZ, RELOCK_MARGIN, TRACKER
from .inference import get_model, pick_my_character


def _video_fps(src: str) -> float:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def track_video(
    src: str,
    *,
    conf: float = 0.4,
    coast_grace: int = COAST_GRACE,
    challenge_frames: int = CHALLENGE_FRAMES,
    relock_margin: float = RELOCK_MARGIN,
    draw: bool = True,
) -> Iterator[dict]:
    """영상을 추적하며 프레임마다 {frame, metrics, fps} 를 yield 한다.

    frame: '내 캐릭터' 박스(MINE)가 강조된 BGR ndarray (draw=True일 때).
    metrics: 현재까지 누적된 추적 지표 스냅샷.
    """
    fps = _video_fps(src)

    locked_id = locked_score = None
    challenger_id = None
    challenger_count = 0
    coast_run = 0
    n = frames_my = any_char = any_uid = id_switches = max_coast = 0

    results = get_model().track(
        source=src, imgsz=IMGSZ, conf=conf, persist=True,
        tracker=str(TRACKER), stream=True, verbose=False,
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
            any_char += 1
        if ids:
            any_uid += 1

        my = pick_my_character(chars, ids)
        cand_tid = my["character"]["tid"] if my else None
        cand_score = my["score"] if my else None

        locked_present = locked_id is not None and locked_id in by_tid
        if locked_present:
            # 락 유지(관성): 다른 후보가 challenge_frames 동안 연속 우세할 때만 교체.
            locked_score = by_tid[locked_id]["conf"]
            if (cand_tid is not None and cand_tid != locked_id and cand_score is not None
                    and cand_score > locked_score + relock_margin):
                if cand_tid == challenger_id:
                    challenger_count += 1
                else:
                    challenger_id, challenger_count = cand_tid, 1
                if challenger_count >= challenge_frames:
                    locked_id, locked_score = challenger_id, cand_score
                    challenger_id, challenger_count = None, 0
                    id_switches += 1
            else:
                challenger_id, challenger_count = None, 0
        else:
            # 락이 없거나 사라짐. 유예기간 내엔 같은 ID 복귀를 기다리며 재획득 보류.
            if locked_id is None or coast_run >= coast_grace:
                if cand_tid is not None:
                    if locked_id is not None and cand_tid != locked_id:
                        id_switches += 1
                    locked_id, locked_score = cand_tid, cand_score
                    challenger_id, challenger_count = None, 0

        my_box = by_tid.get(locked_id) if locked_id is not None else None
        if my_box is not None:
            frames_my += 1
            coast_run = 0
        else:
            coast_run += 1
            max_coast = max(max_coast, coast_run)

        frame = None
        if draw:
            frame = r.plot()
            if my_box is not None:
                cv2.rectangle(frame, (int(my_box["x1"]), int(my_box["y1"])),
                              (int(my_box["x2"]), int(my_box["y2"])), (0, 0, 255), 4)
                cv2.putText(frame, "MINE", (int(my_box["x1"]), max(20, int(my_box["y1"]) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "(my character lost)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        yield {
            "frame": frame,
            "fps": fps,
            "metrics": {
                "frames": n,
                "my_located_rate": round(frames_my / n, 3) if n else 0,
                "any_char_rate": round(any_char / n, 3) if n else 0,
                "any_uid_rate": round(any_uid / n, 3) if n else 0,
                "id_switches": id_switches,
                "max_coast_frames": max_coast,
            },
        }
