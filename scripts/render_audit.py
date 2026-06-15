# 감사 결과 시각화: GT(초록) vs 라벨에 없는 탐지(빨강)
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "labeled"

parser = argparse.ArgumentParser()
parser.add_argument("stems", nargs="+")
parser.add_argument("--weights", default=str(ROOT / "runs/df_yolo11s_1280_e80/weights/best.pt"))
parser.add_argument("--outdir", default=str(ROOT / "reviews" / "review_audit"))
args = parser.parse_args()
WEIGHTS = Path(args.weights)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


model = YOLO(WEIGHTS)
for stem in args.stems:
    img_path = SRC / (stem + ".jpg")
    r = model.predict(source=str(img_path), imgsz=1280, conf=0.5, verbose=False, device=0)[0]
    data = json.loads((SRC / (stem + ".json")).read_text(encoding="utf-8"))
    gt = []
    for s in data["shapes"]:
        if s["label"] == "user_character":
            (x1, y1), (x2, y2) = s["points"]
            gt.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    im = Image.open(img_path)
    d = ImageDraw.Draw(im)
    for g in gt:
        d.rectangle(g, outline="lime", width=4)
    for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        if int(cls) == 0 and all(iou(box, g) < 0.3 for g in gt):
            d.rectangle(box, outline="red", width=4)
            d.text((box[0], box[1] - 14), f"{conf:.2f}", fill="red")
    im.thumbnail((1400, 1400))
    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{stem}.jpg"
    im.save(out, quality=85)
    print(out)
