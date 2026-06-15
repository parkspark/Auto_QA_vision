# 던파 스크린샷 탐지 테스트 웹앱
# v3 모델로 character/user_id를 탐지하고, A안 후처리(user_id 바로 아래 character)로 "내 캐릭터"를 선정한다.
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "df_yolo11s_1280_v3" / "weights" / "best.pt"  # v3: mAP50 0.931 (v2 0.923 대비 개선)

app = Flask(__name__)
model = YOLO(WEIGHTS)


def pick_my_character(chars, ids):
    """가장 신뢰도 높은 user_id 아래에 정렬된 character를 내 캐릭터로 선정"""
    best = None
    for uid in ids:
        ucx = (uid["x1"] + uid["x2"]) / 2
        for ch in chars:
            ccx = (ch["x1"] + ch["x2"]) / 2
            dx = abs(ccx - ucx)
            dy = ch["y1"] - uid["y2"]  # 캐릭터 상단과 닉네임 하단의 수직 간격
            if dy < -40 or dy > 150:   # 닉네임이 캐릭터 머리 위에 있어야 함
                continue
            if dx > (ch["x2"] - ch["x1"]):  # 수평으로 캐릭터 폭 이상 벗어나면 제외
                continue
            score = uid["conf"] - dx / 500 - abs(dy) / 500
            if best is None or score > best["score"]:
                best = {"score": score, "character": ch, "user_id": uid}
    return best


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "image 파일이 없습니다"}), 400
    try:
        conf = max(0.05, min(0.95, float(request.form.get("conf", 0.5))))
    except ValueError:
        conf = 0.5
    im = Image.open(f.stream).convert("RGB")
    r = model.predict(im, imgsz=1280, conf=conf, verbose=False)[0]

    chars, ids = [], []
    for box, cls, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        d = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "conf": round(c, 3)}
        (chars if int(cls) == 0 else ids).append(d)

    my = pick_my_character(chars, ids)
    return jsonify(
        {
            "image": {"width": im.width, "height": im.height},
            "characters": chars,
            "user_ids": ids,
            "my_character": None if my is None else {"character": my["character"], "user_id": my["user_id"]},
        }
    )


if __name__ == "__main__":
    print(f"모델: {WEIGHTS}")
    print("브라우저에서 http://localhost:5000 을 여세요")
    app.run(host="127.0.0.1", port=5000)
