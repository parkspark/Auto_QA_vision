# datasets/df_video의 pseudo-label 영상 프레임을 datasets/df/images(labels)/train에 병합
# convert+synthesize 재실행 '후'에 호출한다. val에는 절대 넣지 않는다(영상 도메인은 train 전용,
# 평가는 사람 라벨 stills val + 07-25 추적지표로 분리).
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "datasets" / "df_video"
TRAIN_IMG = ROOT / "datasets" / "df" / "images" / "train"
TRAIN_LBL = ROOT / "datasets" / "df" / "labels" / "train"


def main():
    TRAIN_IMG.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL.mkdir(parents=True, exist_ok=True)
    n = 0
    for lbl in sorted((VIDEO / "labels").glob("vid_*.txt")):
        img = VIDEO / "images" / (lbl.stem + ".jpg")
        if not img.exists():
            continue
        shutil.copy2(img, TRAIN_IMG / img.name)
        shutil.copy2(lbl, TRAIN_LBL / lbl.name)
        n += 1
    print(f"merged {n} video frames into train")
    total = len(list(TRAIN_IMG.glob('*.jpg')))
    print(f"train images total: {total}")


if __name__ == "__main__":
    main()
