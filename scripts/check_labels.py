# 라벨 박스를 그려 시각 검증용 이미지 생성
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "df"

for name in sys.argv[1:]:
    im = Image.open(DS / "images" / "train" / (name + ".jpg"))
    d = ImageDraw.Draw(im)
    W, H = im.size
    for line in (DS / "labels" / "train" / (name + ".txt")).read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
        d.rectangle([x1, y1, x2, y2], outline=("red" if c == 0 else "cyan"), width=4)
    im.thumbnail((1400, 1400))
    out = ROOT / f"check_{name}.jpg"
    im.save(out, quality=85)
    print(out)
