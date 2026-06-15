# 검수 결과(O 판정 박스)를 원본 labelme JSON에 user_character로 추가
# ? 판정 중 추가 확인으로 O 판정된 box_id는 EXTRA_O에 명시
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "labeled"

parser = argparse.ArgumentParser()
parser.add_argument("--reviewdir", default=str(ROOT / "reviews" / "review"))
parser.add_argument("--result", default="review_result.csv", help="reviewdir 안의 검수 결과 파일명")
parser.add_argument("--extra-o", default="", help="'?' 중 추가 확인으로 O 판정된 box_id (쉼표 구분)")
args = parser.parse_args()
REVIEW = Path(args.reviewdir)
EXTRA_O = {int(x) for x in args.extra_o.split(",") if x.strip()}

verdicts = {}
with (REVIEW / args.result).open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        verdicts[int(row["box_id"])] = row["verdict"].strip()

boxes = json.loads((REVIEW / "boxes.json").read_text(encoding="utf-8"))
to_add = [b for b in boxes if verdicts.get(b["box_id"]) == "O" or b["box_id"] in EXTRA_O]
print(f"adding {len(to_add)} boxes")

by_image = {}
for b in to_add:
    by_image.setdefault(b["image"], []).append(b)

for stem, items in by_image.items():
    jp = SRC / (stem + ".json")
    data = json.loads(jp.read_text(encoding="utf-8"))
    for b in items:
        x1, y1, x2, y2 = b["xyxy"]
        data["shapes"].append(
            {
                "label": "user_character",
                "text": "",
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {"auto_added": True, "conf": b["conf"]},
            }
        )
    jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"updated {len(by_image)} json files")
