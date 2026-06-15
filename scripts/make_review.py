# 검수 패키지 생성:
# 1) 빨간 박스(라벨에 없는 character 탐지)마다 주변 맥락 포함 크롭 이미지 저장
# 2) 박스 좌표 메타데이터(boxes.json) 저장 — 검수 후 라벨 반영에 사용
# 3) 클릭 검수용 review.html 생성
import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "labeled"
CONF = 0.5
IOU_MATCH = 0.3
MARGIN = 80  # 크롭 시 주변 맥락 픽셀

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=str(ROOT / "runs/df_yolo11s_1280_e80/weights/best.pt"))
parser.add_argument("--audit", default=str(ROOT / "outputs" / "unlabeled_audit.csv"))
parser.add_argument("--outdir", default=str(ROOT / "reviews" / "review"))
parser.add_argument("--storage-key", default="df_review_v1")  # 버전별로 분리해야 검수 진행상태가 안 섞임
args = parser.parse_args()
WEIGHTS = Path(args.weights)
AUDIT_CSV = Path(args.audit)
OUT = Path(args.outdir)
CROPS = OUT / "crops"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    CROPS.mkdir(parents=True, exist_ok=True)
    import csv

    stems = []
    with AUDIT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["extra_dets"]) > 0:
                stems.append(row["image"])

    model = YOLO(WEIGHTS)
    boxes_meta = []
    box_id = 0
    for stem in stems:
        img_path = SRC / (stem + ".jpg")
        r = model.predict(source=str(img_path), imgsz=1280, conf=CONF, verbose=False, device=0)[0]
        data = json.loads((SRC / (stem + ".json")).read_text(encoding="utf-8"))
        gt = []
        for s in data["shapes"]:
            if s["label"] == "user_character":
                (x1, y1), (x2, y2) = s["points"]
                gt.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        im = Image.open(img_path)
        W, H = im.size
        for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if int(cls) != 0 or any(iou(box, g) >= IOU_MATCH for g in gt):
                continue
            x1, y1, x2, y2 = box
            crop = im.crop((max(0, x1 - MARGIN), max(0, y1 - MARGIN), min(W, x2 + MARGIN), min(H, y2 + MARGIN)))
            # 크롭 안에 대상 박스 표시
            from PIL import ImageDraw

            d = ImageDraw.Draw(crop)
            ox, oy = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
            d.rectangle((x1 - ox, y1 - oy, x2 - ox, y2 - oy), outline="red", width=3)
            crop_name = f"box{box_id:04d}.jpg"
            crop.save(CROPS / crop_name, quality=85)
            boxes_meta.append(
                {"box_id": box_id, "image": stem, "conf": round(conf, 3), "xyxy": [round(v, 1) for v in box], "crop": f"crops/{crop_name}"}
            )
            box_id += 1

    (OUT / "boxes.json").write_text(json.dumps(boxes_meta, ensure_ascii=False, indent=1), encoding="utf-8")

    cards = "\n".join(
        f'<div class="card" id="c{b["box_id"]}"><img src="{b["crop"]}" loading="lazy">'
        f'<div class="meta">#{b["box_id"]} | {b["image"]}<br>conf {b["conf"]}</div>'
        f'<div class="btns"><button class="o" onclick="mark({b["box_id"]},\'O\')">O 캐릭터</button>'
        f'<button class="x" onclick="mark({b["box_id"]},\'X\')">X 아님</button>'
        f'<button class="q" onclick="mark({b["box_id"]},\'?\')">? 애매</button></div></div>'
        for b in boxes_meta
    )
    html = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>박스 검수</title><style>
body{font-family:sans-serif;background:#1e1e1e;color:#eee;margin:16px}
.top{position:sticky;top:0;background:#1e1e1e;padding:8px 0;z-index:9}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:#2a2a2a;border:3px solid #444;border-radius:8px;padding:8px}
.card img{width:100%;border-radius:4px}
.card.O{border-color:#3c3}.card.X{border-color:#e44}.card.Q{border-color:#fa0}
.meta{font-size:12px;color:#aaa;margin:6px 0}
button{margin-right:6px;padding:6px 10px;border:none;border-radius:4px;cursor:pointer}
.o{background:#2a7}.x{background:#c33}.q{background:#a70}
.big{font-size:15px;padding:8px 14px}
#stat{margin-left:12px}
</style></head><body>
<div class="top">
<button class="big o" onclick="markAll('O')">전부 O로 채우기</button>
<button class="big" onclick="exportCsv()">결과 내보내기 (CSV 다운로드)</button>
<span id="stat"></span>
</div>
<div class="grid">
__CARDS__
</div>
<script>
const KEY='__STORAGE_KEY__';
let st=JSON.parse(localStorage.getItem(KEY)||'{}');
function paint(){
  for(const[id,v]of Object.entries(st)){
    const c=document.getElementById('c'+id);
    if(c){c.classList.remove('O','X','Q');c.classList.add(v==='?'?'Q':v);}
  }
  const n=Object.keys(st).length;
  document.getElementById('stat').textContent=`판정 ${n} / __TOTAL__`;
}
function mark(id,v){st[id]=v;localStorage.setItem(KEY,JSON.stringify(st));paint();}
function markAll(v){document.querySelectorAll('.card').forEach(c=>{st[c.id.slice(1)]=v});localStorage.setItem(KEY,JSON.stringify(st));paint();}
function exportCsv(){
  let rows=['box_id,verdict'];
  document.querySelectorAll('.card').forEach(c=>{const id=c.id.slice(1);rows.push(id+','+(st[id]||''))});
  const blob=new Blob([rows.join('\\n')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review_result.csv';a.click();
}
paint();
</script></body></html>"""
    html = html.replace("__CARDS__", cards).replace("__TOTAL__", str(len(boxes_meta))).replace("__STORAGE_KEY__", args.storage_key)
    (OUT / "review.html").write_text(html, encoding="utf-8")
    print(f"boxes: {len(boxes_meta)}")
    print(OUT / "review.html")


if __name__ == "__main__":
    main()
