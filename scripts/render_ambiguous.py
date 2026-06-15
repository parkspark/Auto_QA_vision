# ? 판정 박스를 넓은 맥락(닉네임 확인 가능)으로 렌더링
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "labeled"
MARGIN = 350

REVIEW = Path(sys.argv[1])  # review 패키지 폴더 (boxes.json 포함)
boxes = json.loads((REVIEW / "boxes.json").read_text(encoding="utf-8"))
ids = [int(x) for x in sys.argv[2:]]
for b in boxes:
    if b["box_id"] not in ids:
        continue
    im = Image.open(SRC / (b["image"] + ".jpg"))
    W, H = im.size
    x1, y1, x2, y2 = b["xyxy"]
    d = ImageDraw.Draw(im)
    d.rectangle((x1, y1, x2, y2), outline="red", width=4)
    crop = im.crop((max(0, x1 - MARGIN), max(0, y1 - MARGIN), min(W, x2 + MARGIN), min(H, y2 + MARGIN)))
    out = REVIEW / f"amb_box{b['box_id']:04d}.jpg"
    crop.save(out, quality=88)
    print(out)
