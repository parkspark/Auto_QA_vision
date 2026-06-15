# v2/v3 보고서용 예시 이미지 생성
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import pick_my_character

ASSETS = ROOT / "reports" / "report_v2_v3_assets"
ASSETS.mkdir(exist_ok=True)
V2 = ROOT / "runs/df_yolo11s_1280_v2/weights/best.pt"
V3 = ROOT / "runs/df_yolo11s_1280_v3/weights/best.pt"
VID = ROOT / "datasets/df/videos/dnfvideo 2025-07-25 00-03-38-255.avi"
VID_EFFECT = ROOT / "datasets/df/videos/dnfvideo 2025-03-22 15-24-45-937.avi"


def banner(img, text, h=44):
    bar = np.full((h, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([bar, img])


def fit(img, w=720):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h))


def fit_h(img, h=430):
    w = int(img.shape[1] * h / img.shape[0])
    return cv2.resize(img, (w, h))


def main():
    from ultralytics import YOLO
    m2, m3 = YOLO(V2), YOLO(V3)

    # ---- Asset A: 데이터 성질 비교 (정적 실사 vs 동영상 프레임) ----
    import json
    still_path = None
    for jp in sorted((ROOT / "labeled").glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        labels = [s["label"] for s in data.get("shapes", [])]
        if labels.count("user_character") >= 2 and "user_id" in labels:
            cand = jp.with_suffix(".jpg")
            if cand.exists():
                still_path = cand
                break
    still = cv2.imread(str(still_path or sorted((ROOT / "labeled").glob("*.jpg"))[10]))
    capE = cv2.VideoCapture(str(VID_EFFECT))
    capE.set(cv2.CAP_PROP_POS_FRAMES, 460)  # 이펙트/블러 많은 보스전 프레임
    _, eff = capE.read()
    capE.release()
    a = np.hstack([banner(fit_h(still), "v2 data: static screenshot (clean)"),
                   banner(fit_h(eff), "v3 added: video frame (blur / effects)")])
    cv2.imwrite(str(ASSETS / "data_nature.png"), a)

    # ---- 07-25에서 다인 파티 프레임 하나 선택 (v3 기준 character>=3) ----
    cap = cv2.VideoCapture(str(VID))
    pick = None
    for idx in range(900, 2600, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        r = m3.predict(fr, imgsz=1280, conf=0.4, verbose=False)[0]
        if int((r.boxes.cls == 0).sum()) >= 3:
            pick = (idx, fr)
            break
    cap.release()
    idx, frame = pick

    # ---- Asset B: v2 vs v3 탐지 비교 (동일 프레임) ----
    p2 = m2.predict(frame, imgsz=1280, conf=0.4, verbose=False)[0].plot()
    p3 = m3.predict(frame, imgsz=1280, conf=0.4, verbose=False)[0].plot()
    b = np.hstack([banner(fit_h(p2), "v2 detection"), banner(fit_h(p3), "v3 detection")])
    cv2.imwrite(str(ASSETS / "det_compare.png"), b)

    # ---- Asset C: v3 추적 + 내 캐릭터 락온 예시 ----
    r = m3.predict(frame, imgsz=1280, conf=0.4, verbose=False)[0]
    chars, ids = [], []
    for box, cls, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        d = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "conf": c}
        (chars if int(cls) == 0 else ids).append(d)
    my = pick_my_character(chars, ids)
    canvas = r.plot()
    if my:
        ch = my["character"]
        cv2.rectangle(canvas, (int(ch["x1"]), int(ch["y1"])), (int(ch["x2"]), int(ch["y2"])), (0, 0, 255), 4)
        cv2.putText(canvas, "MINE", (int(ch["x1"]), max(24, int(ch["y1"]) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imwrite(str(ASSETS / "tracking_example.png"), banner(fit(canvas, 960),
                "v3 + post-processing: red box = estimated 'my character' (lock-on)"))

    print(f"assets saved to {ASSETS} (party frame idx={idx})")


if __name__ == "__main__":
    main()
