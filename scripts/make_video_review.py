# 영상 프레임 pseudo-label 검수 페이지 생성 (이전 make_review.py와 동일 UX)
# datasets/df_video의 각 character(class 0) pseudo-label 박스를 주변 맥락과 함께 크롭하고,
# O(진짜 플레이어) / X(NPC·소환수·오탐) / ?(애매) 클릭 검수용 review.html을 만든다.
# 결과 CSV는 apply 단계에서 X 박스를 라벨에서 제거하는 데 쓸 수 있다.
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "datasets" / "df_video"
MARGIN = 90  # 크롭 시 주변 맥락 픽셀


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "reviews" / "review_video"))
    ap.add_argument("--storage-key", default="df_video_review_v1")
    args = ap.parse_args()
    OUT = Path(args.outdir)
    CROPS = OUT / "crops"
    CROPS.mkdir(parents=True, exist_ok=True)

    boxes_meta = []
    box_id = 0
    for lbl in sorted((VIDEO / "labels").glob("vid_*.txt")):
        img_path = VIDEO / "images" / (lbl.stem + ".jpg")
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) != 5 or p[0] != "0":
                continue  # character 박스만 검수 (user_id 제외)
            cx, cy, bw, bh = (float(v) for v in p[1:])
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
            cl = (max(0, x1 - MARGIN), max(0, y1 - MARGIN), min(W, x2 + MARGIN), min(H, y2 + MARGIN))
            crop = im.crop(cl)
            d = ImageDraw.Draw(crop)
            d.rectangle((x1 - cl[0], y1 - cl[1], x2 - cl[0], y2 - cl[1]), outline="red", width=3)
            crop_name = f"box{box_id:04d}.jpg"
            crop.save(CROPS / crop_name, quality=85)
            boxes_meta.append({
                "box_id": box_id,
                "image": lbl.stem,
                "xyxy": [round(v, 1) for v in (x1, y1, x2, y2)],
                "crop": f"crops/{crop_name}",
            })
            box_id += 1

    (OUT / "boxes.json").write_text(json.dumps(boxes_meta, ensure_ascii=False, indent=1), encoding="utf-8")

    cards = "\n".join(
        f'<div class="card" id="c{b["box_id"]}"><img src="{b["crop"]}" loading="lazy">'
        f'<div class="meta">#{b["box_id"]} | {b["image"][:34]}</div>'
        f'<div class="btns"><button class="o" onclick="mark({b["box_id"]},\'O\')">O 플레이어</button>'
        f'<button class="x" onclick="mark({b["box_id"]},\'X\')">X NPC·소환수</button>'
        f'<button class="q" onclick="mark({b["box_id"]},\'?\')">? 애매</button></div></div>'
        for b in boxes_meta
    )
    html = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>영상 프레임 라벨 검수</title><style>
body{font-family:sans-serif;background:#1e1e1e;color:#eee;margin:16px}
.top{position:sticky;top:0;background:#1e1e1e;padding:8px 0;z-index:9}
.hint{font-size:13px;color:#bbb;margin:4px 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:#2a2a2a;border:3px solid #444;border-radius:8px;padding:8px}
.card img{width:100%;border-radius:4px}
.card.O{border-color:#3c3}.card.X{border-color:#e44}.card.Q{border-color:#fa0}
.meta{font-size:12px;color:#aaa;margin:6px 0}
button{margin-right:6px;padding:6px 10px;border:none;border-radius:4px;cursor:pointer;color:#fff}
.o{background:#2a7}.x{background:#c33}.q{background:#a70}
.big{font-size:15px;padding:8px 14px}
#stat{margin-left:12px}
</style></head><body>
<div class="top">
<button class="big o" onclick="markAll('O')">전부 O로 채우기</button>
<button class="big" onclick="exportCsv()">결과 내보내기 (CSV 다운로드)</button>
<span id="stat"></span>
<div class="hint">기준: 빨간 박스가 <b>닉네임을 달 수 있는 플레이어 캐릭터</b>면 O, NPC·소환수·이펙트·오탐이면 X. 판정은 자동 저장됩니다(브라우저).</div>
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
  document.getElementById('stat').textContent=`판정 ${Object.keys(st).length} / __TOTAL__`;
}
function mark(id,v){st[id]=v;localStorage.setItem(KEY,JSON.stringify(st));paint();}
function markAll(v){document.querySelectorAll('.card').forEach(c=>{st[c.id.slice(1)]=v});localStorage.setItem(KEY,JSON.stringify(st));paint();}
function exportCsv(){
  let rows=['box_id,verdict'];
  document.querySelectorAll('.card').forEach(c=>{const id=c.id.slice(1);rows.push(id+','+(st[id]||''))});
  const blob=new Blob([rows.join('\\n')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='video_review_result.csv';a.click();
}
paint();
</script></body></html>"""
    html = html.replace("__CARDS__", cards).replace("__TOTAL__", str(len(boxes_meta))).replace("__STORAGE_KEY__", args.storage_key)
    (OUT / "review.html").write_text(html, encoding="utf-8")
    print(f"character boxes: {len(boxes_meta)}")
    print(OUT / "review.html")


if __name__ == "__main__":
    main()
