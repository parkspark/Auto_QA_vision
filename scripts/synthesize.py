# 스프라이트 합성 데이터 생성
# train 분할의 실제 스크린샷 위에 DF_character_dataset 스프라이트를 1~4개 합성하고
# 원본 라벨 + 합성 박스(class 0)를 병합한 라벨을 생성한다. val에는 합성하지 않는다.
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
SPRITE_DIR = ROOT / "DF_character_dataset"
DS = ROOT / "datasets" / "df"
TRAIN_IMG = DS / "images" / "train"
TRAIN_LBL = DS / "labels" / "train"
N_SYNTH = 2000
SEED = 42
MAX_IOU = 0.25

rng = random.Random(SEED)


def load_real_char_heights():
    """실제 라벨에서 character 박스의 정규화 높이 분포를 수집"""
    heights = []
    for txt in TRAIN_LBL.glob("*.txt"):
        for line in txt.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "0":
                heights.append(float(parts[4]))
    return heights


def sprite_content(im: Image.Image):
    """알파 기준 실제 내용 영역으로 크롭"""
    alpha = np.array(im.split()[-1])
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main():
    sprites = sorted(SPRITE_DIR.rglob("*.png"))
    bgs = sorted(p for p in TRAIN_IMG.glob("*.jpg") if not p.stem.startswith("synth_"))
    heights = load_real_char_heights()
    print(f"sprites={len(sprites)} backgrounds={len(bgs)} real_char_boxes={len(heights)}")

    for i in range(N_SYNTH):
        bg_path = rng.choice(bgs)
        bg = Image.open(bg_path).convert("RGB")
        W, H = bg.size
        lbl_path = TRAIN_LBL / (bg_path.stem + ".txt")
        lines = lbl_path.read_text().splitlines() if lbl_path.exists() else []

        # 기존 박스(픽셀 좌표)로 겹침 회피
        occupied = []
        for line in lines:
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
            occupied.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

        new_lines = list(lines)
        n_paste = rng.randint(1, 4)
        placed = 0
        for _ in range(n_paste):
            sp = Image.open(rng.choice(sprites)).convert("RGBA")
            sp = sprite_content(sp)
            if sp is None:
                continue
            # 실제 캐릭터 높이 분포에서 샘플 + 15% 지터
            th = rng.choice(heights) * rng.uniform(0.85, 1.15) * H
            scale = th / sp.height
            tw = max(8, int(sp.width * scale))
            th = max(8, int(th))
            sp = sp.resize((tw, th), Image.LANCZOS)
            if rng.random() < 0.5:
                sp = sp.transpose(Image.FLIP_LEFT_RIGHT)
            # 장면에 섞이도록 밝기 약간 변형 (알파 보존)
            a = sp.split()[-1]
            sp = ImageEnhance.Brightness(sp.convert("RGB")).enhance(rng.uniform(0.8, 1.15)).convert("RGBA")
            sp.putalpha(a)

            # 게임 플레이 영역에 배치 (우측 채팅창·하단 UI 회피)
            for _attempt in range(20):
                x = rng.randint(int(0.02 * W), max(1, int(0.70 * W) - tw))
                y = rng.randint(int(0.10 * H), max(1, int(0.80 * H) - th))
                box = (x, y, x + tw, y + th)
                if all(iou(box, o) < MAX_IOU for o in occupied):
                    bg.paste(sp, (x, y), sp)
                    occupied.append(box)
                    cx, cy = (x + tw / 2) / W, (y + th / 2) / H
                    new_lines.append(f"0 {cx:.6f} {cy:.6f} {tw / W:.6f} {th / H:.6f}")
                    placed += 1
                    break

        if placed == 0:
            continue
        name = f"synth_{i:05d}"
        bg.save(TRAIN_IMG / f"{name}.jpg", quality=90)
        (TRAIN_LBL / f"{name}.txt").write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{N_SYNTH}")

    n_imgs = len(list(TRAIN_IMG.glob("*.jpg")))
    print(f"done. train images total: {n_imgs}")


if __name__ == "__main__":
    main()
