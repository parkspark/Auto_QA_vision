"""정적 이미지 탐지 + '내 캐릭터' 선정 (A안 후처리).

기존 webapp/app.py와 scripts/track_video.py에 중복돼 있던 pick_my_character를
단일 정의로 통합한다. 모델은 lazy 싱글톤으로 1회만 로드한다.
"""
from __future__ import annotations

from functools import lru_cache

from ultralytics import YOLO

from .config import IMGSZ, WEIGHTS


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """프로세스당 1회만 모델을 로드한다 (FastAPI lifespan에서 워밍업 호출)."""
    return YOLO(str(WEIGHTS))


def pick_my_character(chars: list[dict], ids: list[dict]) -> dict | None:
    """가장 신뢰도 높은 user_id 바로 아래에 정렬된 character를 '내 캐릭터'로 선정.

    A안: 내 캐릭터와 파티원은 시각적으로 구분 불가하므로, 닉네임(user_id) 위치를
    근거로 그 아래 character를 고른다. (webapp·track_video와 동일 규칙)
    """
    best = None
    for uid in ids:
        ucx = (uid["x1"] + uid["x2"]) / 2
        for ch in chars:
            ccx = (ch["x1"] + ch["x2"]) / 2
            dx = abs(ccx - ucx)
            dy = ch["y1"] - uid["y2"]          # 캐릭터 상단과 닉네임 하단의 수직 간격
            if dy < -40 or dy > 150:           # 닉네임이 캐릭터 머리 위에 있어야 함
                continue
            if dx > (ch["x2"] - ch["x1"]):     # 수평으로 캐릭터 폭 이상 벗어나면 제외
                continue
            score = uid["conf"] - dx / 500 - abs(dy) / 500
            if best is None or score > best["score"]:
                best = {"score": score, "character": ch, "user_id": uid}
    return best


def _boxes_to_lists(result) -> tuple[list[dict], list[dict]]:
    """YOLO Result → (characters, user_ids) 딕셔너리 리스트."""
    chars, ids = [], []
    for box, cls, c in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        d = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "conf": round(c, 3)}
        (chars if int(cls) == 0 else ids).append(d)
    return chars, ids


def detect_image(pil_image, conf: float = 0.5) -> dict:
    """RGB PIL 이미지를 받아 탐지 결과 dict를 반환한다.

    반환: {image:{width,height}, characters[], user_ids[], my_character}
    """
    conf = max(0.05, min(0.95, float(conf)))
    result = get_model().predict(pil_image, imgsz=IMGSZ, conf=conf, verbose=False)[0]
    chars, ids = _boxes_to_lists(result)
    my = pick_my_character(chars, ids)
    return {
        "image": {"width": pil_image.width, "height": pil_image.height},
        "characters": chars,
        "user_ids": ids,
        "my_character": None if my is None else {"character": my["character"], "user_id": my["user_id"]},
    }
