# ver_anchor — "처음에 확정한 내 캐릭터를 끝까지 같은 개체로" 추적하는 락온 방식
#
# 기존 track_video.py 와의 철학 차이:
#   - track_video : 매 프레임 "지금 가장 신뢰도 높은 user_id 아래 캐릭터"를 재선정(+hysteresis로 진동 억제)
#   - track_anchor: 초반 워밍업에서 앵커(track ID)를 1회 확정 → 이후엔 외형/위치/닉네임 단서로
#                   '같은 개체'를 유지. user_id는 상시 주신호가 아니라 재확인·최후 폴백으로만 쓴다.
#
# 4단계: ① 앵커 확정(워밍업) ② 락 ID 따라가기 ③ 소실 시 ReID 재획득 ④ 닉네임 정체성 재확인
#
# 사용:
#   python scripts/track_anchor.py --video "<path>" [--weights best.pt] [--out out.mp4]
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_W = ROOT / "runs" / "df_yolo11s_1280_v3" / "weights" / "best.pt"

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import pick_my_character  # 동일 후처리 재사용


# ---------- 외형/닉네임 단서 ----------

def _clamp_box(d, W, H):
    x1 = max(0, int(d["x1"])); y1 = max(0, int(d["y1"]))
    x2 = min(W, int(d["x2"])); y2 = min(H, int(d["y2"]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def hsv_hist(frame, d):
    """character 크롭의 HSV(H,S) 히스토그램. 스킬 이펙트성 초고휘도/암부 픽셀은 마스킹."""
    H, W = frame.shape[:2]
    bb = _clamp_box(d, W, H)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = ((v > 25) & (v < 235)).astype(np.uint8) * 255
    if int(mask.sum()) < 255 * 20:   # 유효 픽셀이 너무 적으면 무효
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def hist_sim(a, b):
    if a is None or b is None:
        return -1.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def nick_crop(frame, d, size=(96, 24)):
    """user_id 박스 픽셀을 고정 크기 그레이스케일로. OCR 없이 닉네임 이미지 식별자로 사용."""
    H, W = frame.shape[:2]
    bb = _clamp_box(d, W, H)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return g


def nick_sim(a, b):
    """동일 크기 그레이 크롭 간 정규화 상관(밝기/대비 변화에 강함)."""
    if a is None or b is None:
        return -1.0
    res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    return float(res[0, 0])


def center(d):
    return ((d["x1"] + d["x2"]) / 2.0, (d["y1"] + d["y2"]) / 2.0)


def uid_above(box, ids):
    """character 박스 바로 위에 정렬된 user_id (pick_my_character와 동일 기하)."""
    bcx = (box["x1"] + box["x2"]) / 2.0
    bw = box["x2"] - box["x1"]
    best, bestdx = None, None
    for uid in ids:
        ucx = (uid["x1"] + uid["x2"]) / 2.0
        dx = abs(ucx - bcx)
        dy = box["y1"] - uid["y2"]
        if dy < -40 or dy > 150 or dx > bw:
            continue
        if bestdx is None or dx < bestdx:
            best, bestdx = uid, dx
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--out", default=None, help="주석 영상 출력 경로 (생략 시 미출력)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--tracker", default=str(ROOT / "scripts" / "bytetrack_df.yaml"))
    # --- ver_anchor 파라미터 ---
    ap.add_argument("--warmup", type=int, default=45,
                    help="앵커 확정 전 후보 점수를 누적할 초반 프레임 수")
    ap.add_argument("--coast-grace", type=int, default=10,
                    help="앵커 ID가 사라져도 이 프레임만큼은 같은 ID 복귀를 기다림(ReID 보류)")
    ap.add_argument("--reid-thresh", type=float, default=0.45,
                    help="외형 히스토그램 상관이 이 값 이상이어야 재획득 인정")
    ap.add_argument("--nick-thresh", type=float, default=0.45,
                    help="닉네임 크롭 상관이 이 값 이상이면 정체성 일치로 간주")
    ap.add_argument("--motion-radius", type=float, default=0.18,
                    help="재획득 시 마지막 위치 기준 허용 반경(프레임 대각선 비율, coast에 비례 확대)")
    ap.add_argument("--hard-reset", type=int, default=120,
                    help="ReID가 이 프레임만큼 실패하면 user_id 기반으로 앵커 강제 재선정(최후 폴백)")
    ap.add_argument("--ema", type=float, default=0.1, help="외형 템플릿 EMA 갱신 계수")
    ap.add_argument("--update-conf", type=float, default=0.6,
                    help="앵커가 이 신뢰도 이상으로 또렷할 때만 외형 템플릿 갱신(이펙트 오염 방지)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.release()
    diag = (W ** 2 + H ** 2) ** 0.5

    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # 상태
    phase = "warmup"
    warm_score = defaultdict(float)      # tid -> 누적 "내 캐릭터" 점수
    warm_hist = {}                       # tid -> 최근 외형 히스토그램
    warm_nick = {}                       # tid -> 최근 닉네임 크롭
    anchor_id = None
    tmpl_hist = None
    tmpl_nick = None
    last_pos = None
    coast_run = 0

    # 지표
    n = 0
    frames_my = frames_any_char = frames_any_uid = 0
    reid_recoveries = 0                  # 같은 개체를 외형으로 되찾은 횟수(좋은 복구)
    hard_resets = 0                      # user_id 폴백으로 앵커가 강제 교체된 횟수(나쁜 전환)
    nick_confirms = 0                    # 닉네임으로 정체성 재확인된 횟수
    fail_run = 0                         # ReID 연속 실패 프레임
    max_coast = 0

    results = model.track(
        source=args.video, imgsz=args.imgsz, conf=args.conf,
        persist=True, tracker=args.tracker, stream=True, verbose=False,
    )

    for r in results:
        n += 1
        frame = r.orig_img
        chars, ids, by_tid = [], [], {}
        boxes = r.boxes
        tids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        for box, cls, c, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(),
                                    boxes.conf.tolist(), tids):
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

        my = pick_my_character(chars, ids)
        my_box = None
        status = ""

        # ---------- ① 워밍업: 앵커 후보 점수 누적 ----------
        if phase == "warmup":
            if my and my["character"]["tid"] is not None:
                tid = my["character"]["tid"]
                warm_score[tid] += my["score"]
                h = hsv_hist(frame, my["character"])
                if h is not None:
                    warm_hist[tid] = h
                warm_nick[tid] = nick_crop(frame, my["user_id"])
                my_box = my["character"]
            status = f"WARMUP {n}/{args.warmup}"

            if n >= args.warmup and warm_score:
                anchor_id = max(warm_score, key=warm_score.get)
                tmpl_hist = warm_hist.get(anchor_id)
                tmpl_nick = warm_nick.get(anchor_id)
                phase = "tracking"
                coast_run = 0
            elif n >= args.warmup and not warm_score:
                # 워밍업 동안 후보가 한 번도 없었으면 창을 연장
                pass

        # ---------- 추적 단계 ----------
        else:
            present = anchor_id in by_tid
            if present:
                # ② 락 ID 따라가기
                box = by_tid[anchor_id]
                my_box = box
                coast_run = 0
                fail_run = 0
                last_pos = center(box)
                if box["conf"] >= args.update_conf:
                    h = hsv_hist(frame, box)
                    if h is not None:
                        tmpl_hist = h if tmpl_hist is None else \
                            cv2.addWeighted(tmpl_hist, 1 - args.ema, h, args.ema, 0)
                # ④ 닉네임 정체성 재확인
                u = uid_above(box, ids)
                if u is not None and tmpl_nick is not None:
                    s = nick_sim(tmpl_nick, nick_crop(frame, u))
                    if s >= args.nick_thresh:
                        nick_confirms += 1
                        nt = nick_crop(frame, u)
                        if nt is not None:
                            tmpl_nick = nt  # 최신 닉네임 외형으로 갱신
                status = f"MINE id={anchor_id}"
            else:
                coast_run += 1
                max_coast = max(max_coast, coast_run)
                if coast_run <= args.coast_grace:
                    # ByteTrack track_buffer로 같은 ID 복귀 기대 → 보류
                    status = f"coasting id={anchor_id} ({coast_run})"
                else:
                    # ③ 소실 → 외형 ReID 재획득 (닉네임 최고치 재선정이 아님!)
                    radius = args.motion_radius * diag * (1 + 0.04 * (coast_run - args.coast_grace))
                    best, best_s = None, -1.0
                    for ch in chars:
                        if ch["tid"] is None:
                            continue
                        if last_pos is not None:
                            cx, cy = center(ch)
                            if ((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2) ** 0.5 > radius:
                                continue
                        s = hist_sim(tmpl_hist, hsv_hist(frame, ch))
                        # 닉네임이 위에 있으면 일치 보너스
                        u = uid_above(ch, ids)
                        if u is not None and tmpl_nick is not None:
                            ns = nick_sim(tmpl_nick, nick_crop(frame, u))
                            if ns >= args.nick_thresh:
                                s += 0.3
                        if s > best_s:
                            best, best_s = ch, s
                    if best is not None and best_s >= args.reid_thresh:
                        anchor_id = best["tid"]
                        my_box = best
                        coast_run = 0
                        fail_run = 0
                        last_pos = center(best)
                        reid_recoveries += 1
                        status = f"REID id={anchor_id} ({best_s:.2f})"
                    else:
                        fail_run += 1
                        status = f"LOST ({fail_run})"
                        # ④ 최후 폴백: 장시간 실패 시 user_id로 앵커 강제 재선정
                        if fail_run >= args.hard_reset and my and my["character"]["tid"] is not None:
                            anchor_id = my["character"]["tid"]
                            tmpl_hist = hsv_hist(frame, my["character"])
                            tmpl_nick = nick_crop(frame, my["user_id"])
                            my_box = my["character"]
                            last_pos = center(my["character"])
                            coast_run = 0
                            fail_run = 0
                            hard_resets += 1
                            status = f"HARD-RESET id={anchor_id}"

        # 추적 단계에서 내 캐릭터(앵커/재획득/폴백)를 실제로 잡은 프레임만 집계
        if phase == "tracking" and my_box is not None:
            frames_my += 1

        # ---------- 주석 ----------
        if writer is not None:
            vis = r.plot()
            if my_box is not None:
                cv2.rectangle(vis, (int(my_box["x1"]), int(my_box["y1"])),
                              (int(my_box["x2"]), int(my_box["y2"])), (0, 0, 255), 3)
            color = (0, 0, 255) if my_box is not None else (0, 165, 255)
            cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            writer.write(vis)

    if writer is not None:
        writer.release()

    metrics = {
        "method": "ver_anchor",
        "video": Path(args.video).name,
        "weights": Path(args.weights).parent.parent.name,
        "frames": n,
        "my_located_rate": round(frames_my / n, 3) if n else 0,
        "any_char_rate": round(frames_any_char / n, 3) if n else 0,
        "any_uid_rate": round(frames_any_uid / n, 3) if n else 0,
        "reid_recoveries": reid_recoveries,
        "hard_resets": hard_resets,
        "nick_confirms": nick_confirms,
        "max_coast_frames": max_coast,
        "anchor_id_final": anchor_id,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    out_json = ROOT / "outputs" / f"track_metrics_ver_anchor_{Path(args.video).stem[:20]}.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
