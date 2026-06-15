# labelme JSON -> YOLO 형식 변환 + train/val 분할
# 라벨 매핑: user_character -> 0 (character), user_id -> 1 (user_id), 그 외 라벨은 제외하고 로그 출력
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "labeled"
DST = ROOT / "datasets" / "df"
VAL_RATIO = 0.1
SEED = 42

CLASS_MAP = {"user_character": 0, "user_id": 1}
CLASS_NAMES = {0: "character", 1: "user_id"}


def convert_one(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w, h = data["imageWidth"], data["imageHeight"]
    lines, skipped = [], []
    for shape in data["shapes"]:
        if shape["shape_type"] != "rectangle":
            skipped.append(f"{json_path.name}: shape_type={shape['shape_type']}")
            continue
        if shape["label"] not in CLASS_MAP:
            skipped.append(f"{json_path.name}: label={shape['label']}")
            continue
        (x1, y1), (x2, y2) = shape["points"]
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)
        if xmax - xmin < 2 or ymax - ymin < 2:
            skipped.append(f"{json_path.name}: degenerate box {shape['label']}")
            continue
        cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        lines.append(f"{CLASS_MAP[shape['label']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, skipped


def main():
    jsons = sorted(SRC.glob("*.json"))
    pairs = []
    all_skipped = []
    for jp in jsons:
        img = jp.with_suffix(".jpg")
        if not img.exists():
            all_skipped.append(f"{jp.name}: image missing")
            continue
        lines, skipped = convert_one(jp)
        all_skipped.extend(skipped)
        if lines:
            pairs.append((img, lines))

    random.Random(SEED).shuffle(pairs)
    n_val = int(len(pairs) * VAL_RATIO)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        img_dir = DST / "images" / split
        lbl_dir = DST / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lines in items:
            shutil.copy2(img, img_dir / img.name)
            (lbl_dir / (img.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {len(items)} images")

    yaml_path = ROOT / "df_dataset.yaml"
    yaml_path.write_text(
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: character\n  1: user_id\n",
        encoding="utf-8",
    )
    print(f"wrote {yaml_path}")
    if all_skipped:
        print(f"\nskipped {len(all_skipped)} shapes:")
        for s in all_skipped:
            print(" ", s)


if __name__ == "__main__":
    main()
