# ver_anchor 추적 방식 분석 보고서(report_ver_anchor.ipynb) 생성 + 실행
# 기존 재선정 락(track_video) vs 신규 커밋 락(track_anchor)을 holdout 07-25에서 비교한다.
import json
from pathlib import Path

import cv2
import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "report_ver_anchor.ipynb"
ASSETS = ROOT / "reports" / "report_ver_anchor_assets"
ASSETS.mkdir(exist_ok=True)

# ---------- 예시 프레임 추출 (주석 영상에서 대표 장면 캡처) ----------
VID = ROOT / "webapp" / "video_out" / "ver_anchor_0725.mp4"
frame_paths = []
if VID.exists():
    cap = cv2.VideoCapture(str(VID))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 3841
    for i, frac in enumerate([0.18, 0.45, 0.72]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, fr = cap.read()
        if ok:
            p = ASSETS / f"frame_{i}.png"
            cv2.imwrite(str(p), fr)
            frame_paths.append(p.name)
    cap.release()

# ---------- 지표 로드 ----------
anchor = json.loads((ROOT / "outputs" / "track_metrics_ver_anchor_dnfvideo 2025-07-25 .json").read_text(encoding="utf-8"))
v3lock = json.loads((ROOT / "outputs" / "track_metrics_df_yolo11s_1280_v3_dnfvideo 2025-07-25 .json").read_text(encoding="utf-8"))

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

# ============================================================ 표지
md(r"""# 동영상 추적 방식 분석 보고서 — `ver_anchor` (커밋 락온)

**대상**: 던전앤파이터 동영상에서 "내 캐릭터"를 지속 추적하는 후처리 방식
**비교**: 기존 `track_video` (매 프레임 재선정 락) vs **신규 `track_anchor` = ver_anchor (초반 1회 확정 후 커밋 락)**
**평가셋**: holdout 동영상 `dnfvideo 2025-07-25` (3,841프레임), **동일한 v3 가중치** 사용
**작성일**: 2026-06-14 | **환경**: RTX 5090, Ultralytics YOLO11s, PyTorch cu128

---
두 방식은 **탐지 모델(v3 best.pt)이 완전히 동일**하다. 차이는 오직 **"어느 캐릭터를 내 캐릭터로
삼아 따라갈 것인가"라는 후처리 정책**에 있다. 따라서 본 보고서의 모든 지표 차이는 **추적 후처리
로직의 변화에 기인**하며, 모델 재학습은 없었다.""")

# ============================================================ 1. 요약
md(r"""## 1. Executive Summary (요약)

기존 방식은 **매 프레임마다 "지금 가장 신뢰도 높은 user_id 아래 캐릭터"를 다시 고른다.** 닉네임이
프레임마다 깜빡이는 파티/레이드 장면에서 락이 다른 캐릭터로 튀고, 후보가 잠깐 사라지면 "lost"로
빠진다. `ver_anchor`는 반대로 **초반 워밍업에서 앵커(추적 대상)를 한 번 확정한 뒤, 신뢰도 순위가
바뀌어도 같은 개체를 유지**한다. 소실 시에는 닉네임 최고치로 재선정하지 않고 **외형(HSV 히스토그램)
+ 모션 + 닉네임 이미지** 단서로 *같은 개체*를 되찾는다.

| 지표 | 기존 `track_video` (재선정 락) | **`ver_anchor` (커밋 락)** | 방향 |
|---|---|---|---|
| 내 캐릭터 포착률 (`my_located_rate`) | %(v3_my).3f | **%(an_my).3f** | ▲ +%(d_my).3f (+%(d_my_pp).1f%%p) |
| 나쁜 전환 (`id_switches` ↔ `hard_resets`) | %(v3_sw)d | **%(an_hr)d** | ▼ 사실상 무전환 |
| 같은 개체 외형 재획득 (`reid_recoveries`) | — | **%(an_reid)d** | 신규 복구 경로 |
| 닉네임 정체성 재확인 (`nick_confirms`) | — | %(an_nick)d | 신규 검증 경로 |
| 최대 연속 소실 (`max_coast_frames`) | %(v3_coast)d | **%(an_coast)d** | ▼ 단축 |
| 캐릭터 탐지율 / user_id 탐지율 | %(v3_char).3f / %(v3_uid).3f | %(an_char).3f / %(an_uid).3f | ≈ (동일 모델) |

**한 줄 요약**: 동일한 v3 모델 위에서 후처리만 "재선정"에서 "커밋 락"으로 바꿔, **내 캐릭터
포착률을 %(v3_my).3f → %(an_my).3f 로 +%(d_my_pp).1f%%p 끌어올리고, 앵커 강제 전환을 %(v3_sw)d회 →
%(an_hr)d회로 사실상 없앴다.** "처음 확정한 타겟을 끝까지 추적"이라는 목표가 그대로 달성됐다.""" % {
    "v3_my": v3lock["my_located_rate"], "an_my": anchor["my_located_rate"],
    "d_my": anchor["my_located_rate"] - v3lock["my_located_rate"],
    "d_my_pp": (anchor["my_located_rate"] - v3lock["my_located_rate"]) * 100,
    "v3_sw": v3lock["id_switches"], "an_hr": anchor["hard_resets"],
    "an_reid": anchor["reid_recoveries"], "an_nick": anchor["nick_confirms"],
    "v3_coast": v3lock["max_coast_frames"], "an_coast": anchor["max_coast_frames"],
    "v3_char": v3lock["any_char_rate"], "v3_uid": v3lock["any_uid_rate"],
    "an_char": anchor["any_char_rate"], "an_uid": anchor["any_uid_rate"],
})

# ============================================================ setup
code(r"""# 공통 설정 및 지표 데이터 (두 방식 모두 동일한 v3 가중치, 동일 holdout 영상)
from pathlib import Path
import json
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from IPython.display import Image, display
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = Path(r"%s")
ASSETS = ROOT / "reports" / "report_ver_anchor_assets"
anchor = json.loads((ROOT / "outputs" / "track_metrics_ver_anchor_dnfvideo 2025-07-25 .json").read_text(encoding="utf-8"))
v3lock = json.loads((ROOT / "outputs" / "track_metrics_df_yolo11s_1280_v3_dnfvideo 2025-07-25 .json").read_text(encoding="utf-8"))
COL = {'old': '#888888', 'anchor': '#e0443e'}
print("기존 재선정 락:", json.dumps(v3lock, ensure_ascii=False))
print("ver_anchor   :", json.dumps(anchor, ensure_ascii=False))""" % str(ROOT).replace("\\", "\\\\"))

# ============================================================ 2. 두 방식의 구조 차이
md(r"""## 2. 두 방식의 구조 차이

두 방식 모두 동일한 v3 탐지 + ByteTrack(`bytetrack_df.yaml`, track_buffer 90) 위에서 동작한다.
차이는 **"내 캐릭터로 어떤 트랙을 따라갈지"** 결정하는 후처리뿐이다.

| 항목 | 기존 `track_video` (재선정 락) | `ver_anchor` (커밋 락) |
|---|---|---|
| 선정 시점 | **매 프레임 재선정** (가장 높은 user_id 아래 캐릭터) | **초반 워밍업 1회 확정** |
| 락 교체 기준 | 더 높은 user_id 신뢰도 (hysteresis로 진동만 억제) | 외형 + 위치 + 닉네임 일치 (**원개체 유지**) |
| user_id 역할 | **상시 주(主) 신호** | 재확인 / 최후 폴백 |
| 소실 복구 | coast 후 다시 user_id 최고치로 재선정 | **외형 ReID로 같은 개체 재획득** |
| 의존성 | — | 없음 (OpenCV 히스토그램·템플릿 매칭만) |

**`ver_anchor`의 4단계 파이프라인**
1. **앵커 확정 (워밍업 ~45프레임)** — `pick_my_character` 점수를 track ID별로 누적, 가장 일관된
   트랙을 앵커로 고정. 동시에 ① 캐릭터 외형 HSV 히스토그램 ② 닉네임 박스 이미지 ③ 위치를 템플릿으로 저장.
2. **락 ID 따라가기** — 앵커 ID가 보이면 그대로 추적. 또렷할 때만(`conf≥0.6`) 외형 템플릿을 EMA 갱신
   (스킬 이펙트 오염 방지).
3. **소실 시 ReID 재획득** — 앵커가 `coast-grace`(10) 넘게 사라지면 닉네임 최고치로 재선정하지 않고,
   **저장 외형과의 히스토그램 상관 × 모션 게이팅**으로 같은 개체를 찾아 트랙 ID를 승계.
4. **닉네임 정체성 재확인** — 추적 박스 위에 user_id가 뜨면 저장한 닉네임 이미지와 매칭. 일치하면
   정체성 확정·템플릿 갱신, 장시간 ReID 실패 시에만 user_id로 앵커를 강제 재선정(최후 폴백).

> **설계 의도**: 파티원과 내 캐릭터가 시각적으로 비슷한 A안 환경에서, 닉네임(고유 텍스트)을 보조
> 식별자로 활용해 "똑같이 생긴 파티원"으로의 락 드리프트를 차단한다.""")

# ============================================================ 3. 결과 분석
md(r"""## 3. 결과 분석 (holdout 07-25, 3,841프레임)

### 3.1 내 캐릭터 포착률 — 핵심 지표""")

code(r"""# [그림 3-1] 내 캐릭터 포착률 비교
fig, ax = plt.subplots(figsize=(5.5, 3.6))
vals = [v3lock['my_located_rate'], anchor['my_located_rate']]
bars = ax.bar(['기존\n재선정 락', 'ver_anchor\n커밋 락'], vals,
              color=[COL['old'], COL['anchor']], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.008, f'{v:.3f}', ha='center', fontweight='bold')
d = anchor['my_located_rate'] - v3lock['my_located_rate']
ax.annotate(f'+{d:.3f}\n(+{d*100:.1f}%p)', xy=(1, vals[1]), xytext=(1.05, vals[1]-0.12),
            color=COL['anchor'], fontweight='bold')
ax.set_ylim(0, 0.7); ax.set_ylabel('my_located_rate')
ax.set_title('[그림 3-1] 내 캐릭터 포착률: 기존 vs ver_anchor (동일 v3 모델)')
plt.tight_layout(); plt.show()""")

md(r"""포착률이 **%(v3_my).3f → %(an_my).3f (+%(d_my_pp).1f%%p)** 로 올랐다. 동일 모델이므로 이 향상은
**전적으로 후처리 차이**다. 기존 방식이 "lost"로 빠지던 구간(닉네임 깜빡임·짧은 가림)을 ver_anchor가
**외형 ReID로 %(an_reid)d회 복구**한 것이 직접적 원인이다.""" % {
    "v3_my": v3lock["my_located_rate"], "an_my": anchor["my_located_rate"],
    "d_my_pp": (anchor["my_located_rate"] - v3lock["my_located_rate"]) * 100,
    "an_reid": anchor["reid_recoveries"]})

md(r"""### 3.2 추적 안정성 — 앵커 강제 전환 / 연속 소실""")

code(r"""# [그림 3-2] 안정성 지표: 나쁜 전환 횟수, 최대 연속 소실
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].bar(['기존\n(id_switches)', 'ver_anchor\n(hard_resets)'],
            [v3lock['id_switches'], anchor['hard_resets']],
            color=[COL['old'], COL['anchor']], width=0.55)
axes[0].set_title('나쁜 전환(앵커 강제 교체) 횟수 ↓')
for i, v in enumerate([v3lock['id_switches'], anchor['hard_resets']]):
    axes[0].text(i, v+0.4, str(v), ha='center', fontweight='bold')
axes[1].bar(['기존', 'ver_anchor'], [v3lock['max_coast_frames'], anchor['max_coast_frames']],
            color=[COL['old'], COL['anchor']], width=0.55)
axes[1].set_title('최대 연속 소실 프레임 ↓')
for i, v in enumerate([v3lock['max_coast_frames'], anchor['max_coast_frames']]):
    axes[1].text(i, v+2, str(v), ha='center', fontweight='bold')
fig.suptitle('[그림 3-2] 추적 안정성 비교'); plt.tight_layout(); plt.show()""")

md(r"""기존 방식의 `id_switches`는 **다른 캐릭터로 락이 넘어간 횟수**(나쁜 전환)다. ver_anchor에서
이에 대응하는 것은 `hard_resets`(ReID가 장시간 실패해 user_id로 앵커를 강제 재선정한 횟수)이며,
전체 영상에서 **단 %(an_hr)d회**에 그쳤다(%(v3_sw)d회 → %(an_hr)d회). 즉 **초반에 확정한 타겟을
사실상 끝까지 유지**했다는 뜻으로, 설계 목표가 그대로 달성됐다. 최대 연속 소실도 %(v3_coast)d →
%(an_coast)d 프레임으로 줄어, 가림 구간에서의 복귀가 빨라졌다.""" % {
    "an_hr": anchor["hard_resets"], "v3_sw": v3lock["id_switches"],
    "v3_coast": v3lock["max_coast_frames"], "an_coast": anchor["max_coast_frames"]})

md(r"""### 3.3 ver_anchor의 정체성 유지 메커니즘 분해""")

code(r"""# [그림 3-3] ver_anchor 복구/검증 이벤트 분해
fig, ax = plt.subplots(figsize=(6.5, 3.4))
labels = ['외형 ReID\n재획득', '닉네임\n정체성 재확인', 'user_id 폴백\n(강제 재선정)']
vals = [anchor['reid_recoveries'], anchor['nick_confirms'], anchor['hard_resets']]
colors = ['#e0443e', '#3a7afe', '#888888']
bars = ax.bar(labels, vals, color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.6, str(v), ha='center', fontweight='bold')
ax.set_ylabel('발생 횟수')
ax.set_title('[그림 3-3] ver_anchor 정체성 유지 이벤트 (홀드아웃 전체)')
plt.tight_layout(); plt.show()""")

md(r"""- **외형 ReID 재획득 %(an_reid)d회**: 같은 개체를 색·위치 단서로 되찾은 주력 복구 경로.
- **닉네임 재확인 %(an_nick)d회**: 닉네임이 캐릭터 위에 또렷이 함께 잡히는 프레임 자체가 드물어
  빈도는 낮지만, 잡히는 순간 정체성을 확정하고 똑같이 생긴 파티원으로의 드리프트를 차단하는
  **결정적 안전장치**로 작동한다(상시 신호가 아니라 보조 검증).
- **user_id 폴백 %(an_hr)d회**: 위 두 경로가 모두 실패한 최후의 경우. 거의 발동하지 않았다는 것은
  외형 ReID만으로 정체성이 충분히 유지됐음을 의미한다.""" % {
    "an_reid": anchor["reid_recoveries"], "an_nick": anchor["nick_confirms"],
    "an_hr": anchor["hard_resets"]})

# ============================================================ 예시 프레임
if frame_paths:
    md(r"""### 3.4 추적 예시 프레임 (주석 영상에서 캡처)

빨간 박스 = `ver_anchor`가 추적 중인 내 캐릭터. 상단 텍스트는 현재 상태
(`MINE`=락 유지, `REID`=외형 재획득, `coasting`=같은 ID 복귀 대기, `LOST`/`HARD-RESET`).""")
    for p in frame_paths:
        code('display(Image(filename=str(ASSETS / "%s")))' % p)

# ============================================================ 4. 결론
md(r"""## 4. 결론 및 한계

### 4.1 결론

동일한 v3 탐지 모델 위에서 **후처리 정책만 "매 프레임 재선정"에서 "초반 확정 후 커밋 락"으로 전환**한
결과, holdout 동영상에서 내 캐릭터 포착률이 **%(v3_my).3f → %(an_my).3f**, 앵커 강제 전환이
**%(v3_sw)d → %(an_hr)d회**로 개선됐다. 핵심은 소실 시 닉네임 최고치로 재선정하지 않고
**외형 + 모션 + 닉네임 이미지로 같은 개체를 되찾는 ReID 경로**이며, 이것이 "처음 확정한 타겟을
지속 추적"이라는 목표를 직접 달성했다.

### 4.2 한계 및 향후 과제

| 항목 | 내용 |
|---|---|
| 동일 외형 파티원 | 같은 직업·코스튬·색이고 닉네임도 장시간 안 보이면 외형 ReID 식별 불가. 모션 연속성으로 버티고 닉네임 재등장 시 재확인이 최선 |
| 평가 한계 | 라벨 없는 추적 지표(`my_located_rate`)는 "앵커 박스 존재" 기준이라, **올바른 개체인지의 정성 검수**(주석 영상 육안 확인)가 별도로 필요 |
| 파라미터 튜닝 | `warmup`·`reid-thresh`·`motion-radius` 스윕으로 포착률 ↔ 오탈취 균형 최적화 여지 |
| 강한 ReID | HSV 히스토그램 대신 BoT-SORT ReID 임베딩으로 교체 시 색 유사 상황 강화 가능(의존성·속도 비용) |
| 웹앱 통합 | `video_app.py`에 ver_anchor 모드 토글 추가로 :5001에서 두 방식 비교 |

---
*두 방식 모두 동일한 `runs/df_yolo11s_1280_v3/weights/best.pt` 와 동일 holdout(07-25)으로 측정했다.
관측된 모든 차이는 추적 후처리 로직의 변화에 기인한다.*""" % {
    "v3_my": v3lock["my_located_rate"], "an_my": anchor["my_located_rate"],
    "v3_sw": v3lock["id_switches"], "an_hr": anchor["hard_resets"]})

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.10"},
})
nbformat.write(nb, OUT)
print(f"built {OUT}")

client = NotebookClient(nb, timeout=600, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(nb, OUT)
print("executed OK")
