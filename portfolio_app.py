# -*- coding: utf-8 -*-
"""
프로젝트 소개(포트폴리오) 페이지 — MINI_DATA_PROJECT (캐릭터 탐지 & 추적)
- 실행:  .venv\\Scripts\\streamlit run portfolio_app.py
- 라이브 탐지/추적 데모는 Flask 웹앱:
    webapp\\start_webapp.bat        (스크린샷 탐지, http://127.0.0.1:5000)
    webapp\\start_video_webapp.bat  (영상 추적,     http://127.0.0.1:5001)

CLAUDE.md / README.md 의 실제 현황(YOLO11s · imgsz 1280 · v1~v4 · ByteTrack 추적)에 맞춰
이전 dataminiproject 의 portfolio_app.py 를 이식·재작성한 랜딩 페이지.
"""
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="비전 기반 게임 QA 자동화 | 캐릭터 탐지 및 추적",
    page_icon="🎮",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────
# 스타일 (다크 + 네온 테크 톤)
# ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --accent: #00e5a8;
        --accent2: #3b82f6;
        --bg-card: rgba(255,255,255,0.04);
        --border: rgba(255,255,255,0.10);
    }
    .stApp {
        background: radial-gradient(1200px 600px at 15% -10%, rgba(59,130,246,0.18), transparent 60%),
                    radial-gradient(1000px 500px at 90% 0%, rgba(0,229,168,0.14), transparent 55%),
                    #0b0f17;
    }
    .block-container { padding-top: 2.5rem; max-width: 1180px; }

    .hero-title {
        font-size: 2.6rem; font-weight: 800; line-height: 1.2;
        background: linear-gradient(90deg, #fff 10%, var(--accent) 90%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: .4rem;
    }
    .hero-sub { font-size: 1.15rem; color: #aab4c5; font-weight: 500; margin-bottom: 1.2rem; }
    .lead { color: #c7d0de; font-size: 1.02rem; line-height: 1.75; }

    .chip {
        display: inline-block; padding: .32rem .8rem; margin: .2rem .3rem .2rem 0;
        border: 1px solid var(--border); border-radius: 999px;
        background: var(--bg-card); color: #d6dde8; font-size: .82rem; font-weight: 600;
    }
    .chip.accent { border-color: rgba(0,229,168,0.5); color: var(--accent); }

    .card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 16px; padding: 1.25rem 1.35rem; height: 100%;
        transition: transform .15s ease, border-color .15s ease;
    }
    .card:hover { transform: translateY(-3px); border-color: rgba(0,229,168,0.45); }
    .card h3 { color: #fff; font-size: 1.08rem; margin: .2rem 0 .55rem; }
    .card p  { color: #aeb8c7; font-size: .92rem; line-height: 1.6; margin: 0; }
    .card .ico { font-size: 1.6rem; }

    .section-title {
        font-size: 1.5rem; font-weight: 800; color: #fff;
        margin: 2.4rem 0 .4rem; padding-left: .7rem;
        border-left: 4px solid var(--accent);
    }
    .section-kicker { color: var(--accent); font-weight: 700; letter-spacing: .08em; font-size: .78rem; }

    .callout {
        background: linear-gradient(90deg, rgba(0,229,168,0.10), rgba(59,130,246,0.06));
        border: 1px solid rgba(0,229,168,0.30); border-left: 4px solid var(--accent);
        border-radius: 12px; padding: 1rem 1.2rem; color: #d8e2ee; line-height: 1.65;
    }
    .pipe {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: .8rem .6rem; text-align: center; height: 100%;
    }
    .pipe .n  { color: var(--accent); font-weight: 800; font-size: .8rem; }
    .pipe .t  { color: #fff; font-weight: 700; font-size: .92rem; margin: .25rem 0; }
    .pipe .d  { color: #9aa6b6; font-size: .78rem; line-height: 1.4; }
    .arrow { color: var(--accent); font-size: 1.4rem; text-align: center; }

    a, a:visited { color: var(--accent); }
    [data-testid="stMetricValue"] { color: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)


def asset(*parts):
    """존재하는 경우에만 절대경로 반환, 없으면 None"""
    p = os.path.join(BASE_DIR, *parts)
    return p if os.path.exists(p) else None


# ────────────────────────────────────────────────────────────────
# HERO
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-kicker">BLACK-BOX · COMPUTER VISION · GAME QA</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">블랙박스 환경 기반의<br>엔드 유저 체감형 게임 QA 자동화</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">화면(픽셀)만 보고 플레이어블 캐릭터를 실시간으로 탐지·추적하는 “AI 테스터”</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lead">게임 내부 코드·메모리에 전혀 접근하지 않고, <b>오직 화면(픽셀)만 보고</b> '
    '캐릭터(<code>character</code>)와 닉네임(<code>user_id</code>)을 탐지한 뒤, '
    '닉네임 바로 아래의 캐릭터를 “내 캐릭터”로 선정(A안 후처리)하여 <b>동영상에서 실시간으로 추적</b>합니다. '
    '실제 대상은 2D 픽셀 액션 게임 <b>던전앤파이터(Dungeon &amp; Fighter)</b> 입니다.</div>',
    unsafe_allow_html=True,
)

st.write("")
st.markdown(
    '<span class="chip accent">YOLO11s</span>'
    '<span class="chip accent">ByteTrack</span>'
    '<span class="chip accent">Computer Vision</span>'
    '<span class="chip">Black-Box QA</span>'
    '<span class="chip">Active Learning</span>'
    '<span class="chip">PyTorch · CUDA</span>'
    '<span class="chip">Flask · Streamlit</span>',
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("탐지 클래스", "2종", "character · user_id")
c2.metric("현역 모델 정확도", "mAP50 0.931", "v3 · imgsz 1280")
c3.metric("학습 데이터", "스틸 907 + 합성 7,070", "+ 영상 프레임 721")
c4.metric("ID 스위치", "276 → 88", "hysteresis 락 ↓68%")

# ────────────────────────────────────────────────────────────────
# 1. 개요
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">1. 프로젝트 개요</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lead">게임 클라이언트 내부의 소스 코드나 메모리 데이터에 전혀 접근하지 않는 '
    '<b>완전한 블랙박스(Black-Box) 환경</b>에서, 실제 플레이어의 시각(화면) 정보만으로 게임 플레이를 '
    '검증하는 자동화 테스트 시스템입니다. 화려한 이펙트와 UI가 산재한 인게임 화면 속에서 '
    'YOLO11s 모델이 <b>모든 플레이어 캐릭터(character)</b>와 <b>닉네임(user_id)</b>을 분리 탐지하고, '
    '“내 캐릭터”는 학습이 아니라 <b>후처리 정책</b>(가장 신뢰도 높은 닉네임 박스 바로 아래의 캐릭터 선택)으로 '
    '결정합니다. — 내 캐릭터와 파티원은 <i>시각적으로 구분 불가</i>하기 때문입니다.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="callout">🎯 <b>2단계 목표</b> &nbsp;①&nbsp; 정적 스크린샷 탐지 (v1~v2) '
    '&nbsp;→&nbsp; ②&nbsp; <b>동영상에서 캐릭터 실시간 추적</b> (v3~, 현재 단계).</div>',
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# 2. 목적
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">2. 프로젝트 목적</div>', unsafe_allow_html=True)
obj = [
    ("🎯", "엔드 유저 관점의 검증", "내부 API·로그로는 확인하기 힘든, 실제 모니터에 출력되는 렌더링 결과물 기반의 결함(캐릭터 낌, 이펙트 가림, 비정상 좌표 이동 등)을 플레이어 눈높이에서 자동 확인합니다."),
    ("🛡️", "비간섭적(Non-Intrusive) 테스트", "게임 엔진·클라이언트에 영향을 주지 않고 영상 스트림만 분석합니다. 라이브 빌드에도 즉시 적용 가능한 QA 파이프라인입니다."),
    ("♻️", "반복 테스트 피로도 해소", "수동으로 하던 장시간 플레이 모니터링·버그 재현을 비전 모델이 대체하여 회귀 테스트의 효율과 정확도를 극대화합니다."),
]
cols = st.columns(3)
for col, (ico, title, desc) in zip(cols, obj):
    col.markdown(
        f'<div class="card"><div class="ico">{ico}</div><h3>{title}</h3><p>{desc}</p></div>',
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# 3. 핵심 가치
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">3. 핵심 가치 & 기술적 도전</div>', unsafe_allow_html=True)
val = [
    ("🔍", "다양한 상황 속 인식률 최적화", "수천 가지 아바타 조합·스킬 이펙트·반투명 UI가 산재한 2D 픽셀 노이즈 속에서도 대상을 분리. 닉네임(원본 ~70×24px)을 잡기 위해 imgsz=1280 로 학습 해상도를 끌어올렸습니다."),
    ("🔒", "추적 정책의 진동 제어", "진짜 병목은 탐지가 아니라 ‘내 캐릭터 선정 정책’의 진동이었습니다. hysteresis 락(coast-grace·challenge-frames)으로 ID 스위치를 276→88 로 줄였습니다."),
    ("📊", "데이터 기반 품질 지표", "라벨 없는 영상에서도 추적률·ID 스위치·coast 프레임을 산출해 특정 맵·상황의 탐지 실패를 정량 진단합니다."),
]
cols = st.columns(3)
for col, (ico, title, desc) in zip(cols, val):
    col.markdown(
        f'<div class="card"><div class="ico">{ico}</div><h3>{title}</h3><p>{desc}</p></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.markdown(
    '<div class="callout">💡 <b>실측 인사이트</b> — 영상 도메인 적응(v3)으로 동영상 캐릭터 탐지율은 '
    '0.885 → <b>0.888</b> 로 올랐지만, pseudo-label 의 “가려진 파티원 미탐지(FN)·모션블러”로 정적 Recall 은 '
    '소폭 하락했습니다. ‘큰 데이터가 항상 낫다’가 아니라 도메인 트레이드오프를 <b>정량적으로</b> 드러냅니다.</div>',
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# 4. 파이프라인
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">4. 시스템 아키텍처 & 파이프라인</div>', unsafe_allow_html=True)
steps = [
    ("STEP 1", "라벨 변환", "labelme JSON → YOLO 포맷 · train/val seed 42 분할"),
    ("STEP 2", "합성 증강", "직업별 스프라이트 7,070장으로 합성 프레임 생성"),
    ("STEP 3", "영상 프레임 병합", "동영상 0.5fps 추출 → pseudo-label → train 병합"),
    ("STEP 4", "모델 학습", "YOLO11s · imgsz 1280 · 버전별(v1~v4) 학습"),
    ("STEP 5", "추적·평가", "ByteTrack + hysteresis 락 추적 · 추적지표 산출"),
]
cols = st.columns(len(steps) * 2 - 1)
for i, (n, t, d) in enumerate(steps):
    with cols[i * 2]:
        st.markdown(f'<div class="pipe"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>', unsafe_allow_html=True)
    if i < len(steps) - 1:
        cols[i * 2 + 1].markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)

st.write("")
st.caption(
    "능동학습 루프: 학습된 모델이 새 스틸/영상 프레임을 1차 라벨링(pseudo-label)하고, 사람은 "
    "*처음부터 그리는 대신 O/X/? 검수만* 합니다. NPC·소환수 오탐 107건을 제거(v4 데이터셋)하는 식으로 "
    "데이터가 쌓일수록 라벨링 비용은 줄고 모델은 강해지는 선순환을 구현했습니다."
)

with st.expander("📂 파이프라인 스크립트 구성 보기 (scripts/)"):
    st.markdown(
        """
| 단계 | 스크립트 | 역할 |
|------|----------|------|
| 라벨 변환·분할 | `convert_labelme.py` | labelme JSON → YOLO 라벨 + train/val 분할(seed 42) |
| 합성 증강 | `synthesize.py` | 직업별 스프라이트 합성으로 학습 프레임 2,000장 생성 |
| 영상 프레임 추출 | `extract_frames.py` | 동영상 0.5fps 추출 + v2/v3 pseudo-label(conf 0.5 게이팅) |
| 영상 프레임 병합 | `merge_video_frames.py` | 영상 프레임을 train 셋에 병합 (val 오염 금지) |
| 모델 학습 | `train.py` / `resume_train.py` | YOLO11s · imgsz 1280 학습 · 중단 복구(resume) |
| 영상 추적·평가 | `track_video.py` | ByteTrack + hysteresis 락 추적, 추적지표 JSON 산출 |
| 라벨 검수 | `make_review.py` · `apply_review.py` | 클릭 O/X/? 검수 페이지 생성 → 원본 라벨 반영(백업 먼저) |

> ⚙️ Windows 워커 안정성을 위해 `cache="disk"`(.npy) + `workers` 튜닝으로 RAM 캐시 직렬화 크래시를 회피했습니다.
        """
    )

# ────────────────────────────────────────────────────────────────
# 5. 쇼케이스
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">5. 탐지 & 추적 쇼케이스</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🎬 영상 추적 데모", "🆚 버전별 탐지 비교", "🧩 데이터 & 추적 예시"])

with tab1:
    v3 = asset("outputs", "track_v3_hysteresis_0725.mp4")
    v2 = asset("outputs", "track_baseline_v2_0725.mp4")
    if v3 or v2:
        vc1, vc2 = st.columns(2)
        if v3:
            with vc1:
                st.video(v3)
                st.caption("v3 + hysteresis 락 — ID 스위치 23, 안정적인 ‘내 캐릭터’ 락온")
        if v2:
            with vc2:
                st.video(v2)
                st.caption("v2 baseline — 정책 진동으로 ID 스위치 263 (락 적용 전)")
        st.caption("2025-07-25 holdout 영상(3,841 프레임)에서의 추적 결과. 라벨 없이 추적률·ID 스위치를 산출합니다.")
    else:
        st.info("추적 영상을 찾을 수 없습니다. (outputs/ 폴더 확인)")

with tab2:
    p = asset("reports", "report_v2_v3_assets", "det_compare.png")
    if p:
        st.image(p, caption="버전별 탐지 결과 비교 (v2 vs v3)", use_container_width=True)
    else:
        st.info("비교 이미지를 찾을 수 없습니다. (reports/report_v2_v3_assets/ 확인)")
    st.caption(
        "v3 는 영상 프레임 721장(pseudo-label)을 추가해 동영상 도메인에 적응시킨 현역 모델입니다 (val mAP50 0.931)."
    )

with tab3:
    imgs, caps = [], []
    for fn, cap in [
        ("data_nature.png", "데이터 분포·특성 분석"),
        ("tracking_example.png", "추적 프레임 예시 (character · user_id 박스)"),
    ]:
        p = asset("reports", "report_v2_v3_assets", fn)
        if p:
            imgs.append(p); caps.append(cap)
    if imgs:
        st.image(imgs, caption=caps, use_container_width=True)
    else:
        st.info("예시 이미지를 찾을 수 없습니다.")

# ────────────────────────────────────────────────────────────────
# 6. 버전 히스토리
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">6. 버전 히스토리 & 성능 추이</div>', unsafe_allow_html=True)
st.markdown(
    """
| 버전 | 내용 | val mAP50 |
|------|------|-----------|
| v1 | 실사 817 + 합성 2,000 | 0.856 |
| v2 | 1·2차 라벨 검수 반영 재학습 (character P 0.923) | 0.923 |
| **v3 (현재 운영모델)** | **영상 프레임 721장(pseudo-label) 추가 · 영상 도메인 적응** | **0.931** |
| v4 (데이터 준비됨) | 영상 프레임 검수(NPC 오탐 107 제거) 반영 · 학습 대기 | — |
"""
)
st.caption(
    "v2→v3 트레이드오프: 동영상 추적률 0.840→0.858 · 캐릭터 탐지율 0.885→0.888(↑) ↔ "
    "정적 mAP50-95 0.591→0.573 · Recall 0.858→0.834(↓). 상세는 reports/report_v2_v3.ipynb."
)

# ────────────────────────────────────────────────────────────────
# 7. 엔지니어링 회복력 (하드웨어)
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">7. 불안정한 하드웨어 위의 학습 — “청크 그라인딩”</div>', unsafe_allow_html=True)
hw_img = asset("reports", "report_hardware_assets", "timeline_crash_resume.png") \
    or asset("reports", "report_hardware_assets", "diagram_resume_flow.png")
hc1, hc2 = st.columns([3, 2])
with hc1:
    st.markdown(
        '<div class="callout"><b>문제</b> — GPU 풀로드(학습) 시 전원이 순간 차단되어 강제 재부팅(정전형 크래시). '
        '전력제한·클럭고정으로도 못 막음.<br><br>'
        '<b>해결</b> — 2~8 epoch 단위로 끊어 학습 후 epoch 경계에서 정지 → 가중치·옵티마이저·EMA·LR 스케줄을 '
        'resume 로 복원해 다시 이어붙이는 <b>청크 그라인딩</b>으로 80 epoch 완주에 성공했습니다. '
        '워커 좀비를 트리킬로 정리하고, GPU 메모리가 idle 로 떨어진 뒤 재실행하는 절차까지 정립했습니다. '
        '<b>모델 품질에는 영향 없음</b> — 비용은 재시작 오버헤드뿐입니다.</div>',
        unsafe_allow_html=True,
    )
with hc2:
    if hw_img:
        st.image(hw_img, use_container_width=True)
    else:
        st.info("하드웨어 회복력 다이어그램을 찾을 수 없습니다.")

# ────────────────────────────────────────────────────────────────
# 8. 왜 Vision인가
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">8. 왜 API가 아닌 ‘화면 인식(Vision)’인가?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="callout"><b>블랙박스 테스트의 본질 — “실제 유저와 100% 동일한 환경”</b><br><br>'
    '내부 API·로그·메모리 훅은 게임이 <i>내부적으로</i> 무엇을 하는지 검증합니다. 하지만 '
    '<b>유저가 실제로 보는 것은 모니터에 그려진 최종 픽셀</b>입니다. '
    '내부 좌표는 정상인데 렌더링 단계에서 캐릭터가 UI에 가려지거나, 이펙트가 캐릭터를 덮거나, '
    '프레임이 늦게 그려지는 “체감형 결함”은 오직 화면을 직접 보는 방식으로만 잡을 수 있습니다. '
    '게임 엔진·안티치트에 손대지 않으므로 <b>보안이 철저한 라이브 빌드에도 그대로 적용</b>됩니다.</div>',
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# 9. 기대효과 / 향후
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">9. 기대 효과 & 향후 확장</div>', unsafe_allow_html=True)
colA, colB = st.columns(2)
colA.markdown(
    '<div class="card"><h3>🤖 기대 효과</h3><p> <b>사람의 눈을 대신해 화면을 보고 문제를 진단하는 AI 테스터</b>입니다. '
    '업데이트 시 발생할 수 있는 예기치 못한 사이드 이펙트를 실제 유저가 악용하기 전에 탐지합니다.</p></div>',
    unsafe_allow_html=True,
)
colB.markdown(
    '<div class="card"><h3>🚀 향후 확장</h3><p>'
    '• v4 학습(PSU 교체 후) · pseudo-label 검수 추가로 가려진 파티원 recall 보강<br>'
    '• 모션블러 프레임 정제 · 프레임별 위치·신뢰도 기반 <b>렌더링 지연 정량 리포트</b><br></p></div>',
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# CTA / Footer
# ────────────────────────────────────────────────────────────────
st.write("")
st.markdown(
    '<div class="callout">▶ <b>라이브 데모 실행</b><br>'
    '• 스크린샷 탐지: <code>webapp\\start_webapp.bat</code> → <code>http://127.0.0.1:5000</code> '
    '(이미지 업로드 시 v3 모델이 character·user_id 를 탐지하고 “내 캐릭터”를 선정)<br>'
    '• 영상 추적: <code>webapp\\start_video_webapp.bat</code> → <code>http://127.0.0.1:5001</code> '
    '(동영상에서 ByteTrack + hysteresis 락으로 캐릭터를 추적)</div>',
    unsafe_allow_html=True,
)
st.write("")
st.caption("Vision-based Game QA Automation · Character Tracking · YOLO11s + ByteTrack + Streamlit")
