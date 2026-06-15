# 학습된 모델로 labeled 원본 907장을 재추론해서
# "라벨에 없는데 캐릭터로 탐지된 박스"(미라벨 파티원 후보)가 있는 장면을 찾는다.
import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "labeled"
CONF = 0.5
IOU_MATCH = 0.3

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=str(ROOT / "runs/df_yolo11s_1280_e80/weights/best.pt"))
parser.add_argument("--out", default=str(ROOT / "outputs" / "unlabeled_audit.csv"))
args = parser.parse_args()
WEIGHTS = Path(args.weights)
OUT_CSV = Path(args.out)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    model = YOLO(WEIGHTS)
    rows = []
    for r in model.predict(source=str(SRC), imgsz=1280, conf=CONF, stream=True, verbose=False, device=0):
        stem = Path(r.path).stem
        jp = SRC / (stem + ".json")
        gt = []
        if jp.exists():
            data = json.loads(jp.read_text(encoding="utf-8"))
            for s in data["shapes"]:
                if s["label"] == "user_character":
                    (x1, y1), (x2, y2) = s["points"]
                    gt.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        extra = 0
        for box, cls in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            if int(cls) == 0 and all(iou(box, g) < IOU_MATCH for g in gt):
                extra += 1
        rows.append({"image": stem, "gt_chars": len(gt), "extra_dets": extra})

    rows.sort(key=lambda x: -x["extra_dets"])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "gt_chars", "extra_dets"])
        w.writeheader()
        w.writerows(rows)

    flagged = [r for r in rows if r["extra_dets"] > 0]
    print(f"total={len(rows)} flagged={len(flagged)} extra_boxes={sum(r['extra_dets'] for r in flagged)}")
    print("top 10:")
    for r in flagged[:10]:
        print(f"  {r['image']}: gt={r['gt_chars']} extra={r['extra_dets']}")


if __name__ == "__main__":
    main()
