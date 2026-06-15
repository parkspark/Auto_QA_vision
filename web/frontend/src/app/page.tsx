import Link from "next/link";
import { Section, Stat } from "@/components/ui";
import { VersionChart } from "@/components/version-chart";
import { DemoGallery } from "@/components/demo-gallery";

const TECH = ["YOLO11s", "ByteTrack", "Hysteresis Lock", "FastAPI", "Next.js", "imgsz 1280"];

const PIPELINE = [
  { t: "라벨 변환", d: "labelme JSON → YOLO 포맷 + train/val 분할 (seed 42)" },
  { t: "스프라이트 합성", d: "직업별 단독 스프라이트 7,070장으로 합성 2,000장 생성" },
  { t: "영상 프레임 병합", d: "동영상 0.5fps 추출 + pseudo-label을 train에 병합" },
  { t: "학습", d: "YOLO11s · imgsz 1280 · 80 epoch (전원 이슈로 청크 그라인딩)" },
  { t: "추적", d: "ByteTrack + hysteresis 락으로 '내 캐릭터' 락온" },
];

export default function Home() {
  return (
    <>
      {/* ----------------------------------------------------------- HERO */}
      <section className="mx-auto max-w-6xl px-5 pt-20 pb-12">
        <div className="animate-fade-up">
          <div className="flex flex-wrap gap-2 mb-6">
            {TECH.map((t) => (
              <span key={t} className="chip">{t}</span>
            ))}
          </div>
          <h1 className="text-4xl sm:text-6xl font-bold leading-[1.1] max-w-3xl">
            던전앤파이터 화면에서 <span className="text-gradient">내 캐릭터</span>를 찾아 추적합니다
          </h1>
          <p className="mt-6 text-lg text-muted max-w-2xl">
            정적 스크린샷 탐지부터 동영상 실시간 추적까지. YOLO11s로 캐릭터와 닉네임을 탐지하고,
            후처리·hysteresis 락으로 파티 난전 속에서도 내 캐릭터를 안정적으로 따라갑니다.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/detect" className="btn-brand px-5 py-3 text-sm">라이브 탐지 →</Link>
            <Link href="/track" className="btn-ghost px-5 py-3 text-sm">영상 추적 →</Link>
          </div>
        </div>
      </section>

      {/* ----------------------------------------------------------- STATS */}
      <Section className="!py-8">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <Stat value="0.931" label="val mAP50" sub="v3 · imgsz 1280" />
          <Stat value="276 → 88" label="ID 스위치 −68%" sub="hysteresis 락" />
          <Stat value="2" label="탐지 클래스" sub="character · user_id" />
          <Stat value="8,698" label="학습 데이터" sub="907 실사 + 7,070 합성 + 721 프레임" />
        </div>
      </Section>

      {/* ----------------------------------------------------------- A안 */}
      <Section eyebrow="설계" title="왜 '모두 탐지 후 후처리'인가 (A안)">
        <div className="grid md:grid-cols-2 gap-5">
          <div className="card p-6">
            <h3 className="font-semibold mb-2 text-char">문제</h3>
            <p className="text-sm text-muted leading-relaxed">
              내 캐릭터와 파티원은 시각적으로 구분이 불가능합니다. 모델이 &quot;내 캐릭터만&quot; 직접
              학습하도록 만들 수 없습니다.
            </p>
          </div>
          <div className="card p-6">
            <h3 className="font-semibold mb-2 text-mine">해결</h3>
            <p className="text-sm text-muted leading-relaxed">
              모델은 <span className="text-foreground">모든 캐릭터</span>를 탐지하고, &quot;내 캐릭터&quot;
              선정은 후처리가 담당합니다 — 가장 신뢰도 높은 닉네임(user_id) 바로 아래의 character를 선택.
            </p>
          </div>
        </div>
      </Section>

      {/* ----------------------------------------------------------- PIPELINE */}
      <Section eyebrow="파이프라인" title="라벨에서 추적까지">
        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {PIPELINE.map((s, i) => (
            <div key={s.t} className="card p-5 relative">
              <div className="text-brand font-mono text-sm mb-2">0{i + 1}</div>
              <h3 className="font-semibold mb-1">{s.t}</h3>
              <p className="text-xs text-muted leading-relaxed">{s.d}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* ----------------------------------------------------------- RESULTS */}
      <Section eyebrow="결과" title="버전별 성능 추이">
        <div className="grid lg:grid-cols-2 gap-6 items-center">
          <div className="card p-6">
            <VersionChart />
          </div>
          <div className="space-y-4">
            <p className="text-muted leading-relaxed">
              실사 데이터와 검수 반영, 영상 도메인 적응을 거치며 정적 정확도(mAP50)가 꾸준히 올랐습니다.
              v3은 영상 프레임 721장을 추가해 동영상 도메인에 적응한 현역 모델입니다.
            </p>
            <ul className="space-y-2 text-sm">
              <li className="flex gap-2"><span className="text-brand">▹</span> v1 → v2: 라벨 검수 2사이클 반영, character 정밀도 0.923</li>
              <li className="flex gap-2"><span className="text-brand">▹</span> v2 → v3: 영상 프레임 추가로 추적률 0.840 → 0.858</li>
              <li className="flex gap-2"><span className="text-brand">▹</span> 진짜 병목은 탐지가 아니라 &quot;내 캐릭터 선정 정책&quot;의 진동이었음</li>
            </ul>
          </div>
        </div>
      </Section>

      {/* ----------------------------------------------------------- DEMO */}
      <Section eyebrow="추적 데모" title="Hysteresis 락의 효과">
        <DemoGallery />
      </Section>

      {/* ----------------------------------------------------------- HARDWARE */}
      <Section eyebrow="엔지니어링 노트" title="전원 불안정을 청크 학습으로 우회">
        <div className="card p-6 md:p-8">
          <p className="text-muted leading-relaxed max-w-3xl">
            학습 PC가 GPU 풀로드 시 전원이 순간 차단되어 강제 재부팅되는 문제(Kernel-Power 41)가 있었습니다.
            전력제한·클럭고정으로도 막지 못해, <span className="text-foreground">2~8 epoch 단위로 학습 →
            epoch 경계에서 정지 → resume 반복</span>하는 &quot;청크 그라인딩&quot;으로 80 epoch을 완주했습니다.
            resume가 가중치·옵티마이저·EMA·LR 스케줄을 복원하므로 모델 품질에는 영향이 없고, 비용은 재시작
            오버헤드뿐입니다.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            <span className="chip">Kernel-Power 41</span>
            <span className="chip">resume_train.py</span>
            <span className="chip">epoch 경계 정지</span>
            <span className="chip">80 epoch 완주</span>
          </div>
        </div>
      </Section>

      {/* ----------------------------------------------------------- CTA */}
      <Section className="!pb-24">
        <div className="card p-8 md:p-10 text-center">
          <h2 className="text-2xl sm:text-3xl font-bold mb-3">직접 돌려보세요</h2>
          <p className="text-muted mb-6">스크린샷을 올려 탐지하거나, 짧은 영상으로 추적을 확인할 수 있습니다.</p>
          <div className="flex flex-wrap gap-3 justify-center">
            <Link href="/detect" className="btn-brand px-5 py-3 text-sm">라이브 탐지 →</Link>
            <Link href="/track" className="btn-ghost px-5 py-3 text-sm">영상 추적 →</Link>
          </div>
        </div>
      </Section>
    </>
  );
}
