# 영상 프레임 검수 결과(video_review_result.csv)를 pseudo-label에 반영한다.
# X(NPC·소환수·오탐) 판정된 character 박스를 datasets/df_video/labels에서 제거한다.
# box_id는 make_video_review.py와 동일하게 'sorted 라벨 파일 × 파일 내 character 박스 순서'로 매겨진다.
# user_id(class 1) 박스와 O/?(기본 유지) 박스는 건드리지 않는다. 반영 전 백업 zip 생성.
import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "datasets" / "df_video"
LBL = VIDEO / "labels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(ROOT / "reviews" / "review_video" / "video_review_result.csv"))
    ap.add_argument("--drop-ambiguous", action="store_true", help="?(애매)도 함께 제거 (기본: 유지)")
    ap.add_argument("--dry-run", action="store_true", help="실제 수정 없이 변경 요약만 출력")
    args = ap.parse_args()

    verdict = {}
    with open(args.result, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verdict[int(row["box_id"])] = row["verdict"].strip()

    drop = {"X"} | ({"?"} if args.drop_ambiguous else set())

    # 백업
    if not args.dry_run:
        backup = ROOT / "backups" / f"df_video_labels_backup_{datetime.now():%Y%m%d_%H%M%S}.zip"
        shutil.make_archive(str(backup.with_suffix("")), "zip", LBL)
        print(f"백업: {backup}")

    box_id = 0
    removed = 0
    touched_files = 0
    for lbl in sorted(LBL.glob("vid_*.txt")):
        lines = lbl.read_text().splitlines()
        out = []
        changed = False
        for line in lines:
            p = line.split()
            if len(p) == 5 and p[0] == "0":
                # 이 character 박스의 box_id 결정
                if verdict.get(box_id) in drop:
                    removed += 1
                    changed = True
                    box_id += 1
                    continue  # 제거
                box_id += 1
            out.append(line)
        if changed:
            touched_files += 1
            if not args.dry_run:
                lbl.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

    total_char = box_id
    print(f"검수 박스(character): {total_char}")
    print(f"제거 대상({'/'.join(sorted(drop))}): {removed}개  /  영향 파일: {touched_files}")
    if args.dry_run:
        print("[dry-run] 실제 변경 없음. --dry-run 없이 다시 실행하면 반영됩니다.")
    else:
        print("반영 완료. 다음: datasets/df 재생성(convert+synthesize+merge) → v4 학습.")


if __name__ == "__main__":
    main()
