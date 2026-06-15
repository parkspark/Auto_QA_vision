# CSV 내보내기 없이 '페이지 저장(HTML)'만 한 검수 결과에서 판정을 복구한다.
# paint()가 각 카드에 적용한 클래스(card O/X/Q)가 저장된 DOM에 남아 있으므로 이를 파싱한다.
# 사용: python scripts/recover_review_csv.py <saved.html> [--out result.csv]
import argparse
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("html")
ap.add_argument("--out", default=None)
args = ap.parse_args()

html = Path(args.html).read_text(encoding="utf-8")
# <div class="card O" id="c123"> 형태 (class 안에 O/X/Q 0~1개)
pat = re.compile(r'class="card\s*([OXQ]?)"\s+id="c(\d+)"')
CLASS2V = {"O": "O", "X": "X", "Q": "?", "": ""}

rows = {}
for m in pat.finditer(html):
    verdict = CLASS2V.get(m.group(1), "")
    rows[int(m.group(2))] = verdict

out = Path(args.out) if args.out else Path(args.html).with_name("video_review_result.csv")
lines = ["box_id,verdict"]
for bid in sorted(rows):
    lines.append(f"{bid},{rows[bid]}")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")

from collections import Counter
cnt = Counter(rows.values())
print(f"복구된 박스: {len(rows)}")
print(f"  O(플레이어): {cnt.get('O',0)}  X(NPC/소환수/오탐): {cnt.get('X',0)}  ?(애매): {cnt.get('?',0)}  미판정: {cnt.get('',0)}")
print(f"저장: {out}")
