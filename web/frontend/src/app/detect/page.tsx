"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE, Box, DetectResult, detectImage, detectSample } from "@/lib/api";

const COLORS = { char: "#ef4444", uid: "#22d3ee", mine: "#22c55e" };

interface Sample {
  name: string;
  url: string;
}

export default function DetectPage() {
  const [conf, setConf] = useState(0.4);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/samples/images?limit=6`)
      .then((r) => r.json())
      .then((d) => setSamples(d.images ?? []))
      .catch(() => {});
  }, []);

  const runUpload = useCallback(
    async (file: File) => {
      setError(null);
      setLoading(true);
      setResult(null);
      setPreview(URL.createObjectURL(file));
      try {
        setResult(await detectImage(file, conf));
      } catch (e) {
        setError(e instanceof Error ? e.message : "탐지 실패");
      } finally {
        setLoading(false);
      }
    },
    [conf]
  );

  const runSample = useCallback(
    async (s: Sample) => {
      setError(null);
      setLoading(true);
      setResult(null);
      setPreview(`${API_BASE}${s.url}`);
      try {
        setResult(await detectSample(s.name, conf));
      } catch (e) {
        setError(e instanceof Error ? e.message : "탐지 실패");
      } finally {
        setLoading(false);
      }
    },
    [conf]
  );

  // 박스를 이미지 natural 크기 기준 퍼센트로 환산 → 반응형 오버레이
  const pct = (b: Box, W: number, H: number) => ({
    left: `${(b.x1 / W) * 100}%`,
    top: `${(b.y1 / H) * 100}%`,
    width: `${((b.x2 - b.x1) / W) * 100}%`,
    height: `${((b.y2 - b.y1) / H) * 100}%`,
  });

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="mb-8">
        <p className="text-brand text-sm font-medium uppercase tracking-wide mb-2">라이브 탐지</p>
        <h1 className="text-3xl font-bold">스크린샷 캐릭터 / 닉네임 탐지</h1>
        <p className="text-muted mt-2">
          이미지를 올리면 v3 모델이 <span style={{ color: COLORS.char }}>character</span> ·{" "}
          <span style={{ color: COLORS.uid }}>user_id</span>를 탐지하고, 후처리로{" "}
          <span style={{ color: COLORS.mine }}>내 캐릭터</span>를 선정합니다.
        </p>
      </header>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 컨트롤 */}
        <div className="space-y-5">
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = e.dataTransfer.files?.[0];
              if (f) runUpload(f);
            }}
            className={`card grid place-items-center text-center p-8 cursor-pointer transition-colors ${
              dragOver ? "border-brand bg-brand/5" : ""
            }`}
          >
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && runUpload(e.target.files[0])}
            />
            <div className="text-4xl mb-2">🖼️</div>
            <p className="font-medium">이미지를 드래그하거나 클릭</p>
            <p className="text-xs text-muted mt-1">PNG · JPG</p>
          </label>

          <div className="card p-5">
            <div className="flex justify-between text-sm mb-2">
              <span>신뢰도 임계값</span>
              <span className="text-brand font-mono">{conf.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={conf}
              onChange={(e) => setConf(parseFloat(e.target.value))}
              className="w-full accent-[var(--brand)]"
            />
          </div>

          {samples.length > 0 && (
            <div className="card p-5">
              <p className="text-sm mb-3 text-muted">예시로 시도</p>
              <div className="grid grid-cols-3 gap-2">
                {samples.map((s) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    key={s.name}
                    src={`${API_BASE}${s.url}`}
                    alt={s.name}
                    onClick={() => runSample(s)}
                    className="aspect-video object-cover rounded-md cursor-pointer hover:ring-2 ring-brand transition"
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 이미지 + 박스 오버레이 / 결과 */}
        <div className="lg:col-span-2 space-y-5">
          <div className="card p-3 min-h-[280px] grid place-items-center relative overflow-hidden">
            {loading && (
              <div className="absolute inset-0 grid place-items-center bg-background/60 z-20 rounded-2xl">
                <span className="text-brand animate-pulse">추론 중…</span>
              </div>
            )}
            {error && <p className="text-char text-sm">{error}</p>}
            {!preview && !error && <p className="text-muted text-sm">결과가 여기에 표시됩니다</p>}

            {preview && (
              <div className="relative w-full leading-none">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={preview} alt="입력" className="w-full rounded-lg block" />
                {result &&
                  result.characters.map((b, i) => (
                    <Overlay key={`c${i}`} style={pct(b, result.image.width, result.image.height)} color={COLORS.char} label={`char ${b.conf}`} />
                  ))}
                {result &&
                  result.user_ids.map((b, i) => (
                    <Overlay key={`u${i}`} style={pct(b, result.image.width, result.image.height)} color={COLORS.uid} label={`id ${b.conf}`} />
                  ))}
                {result?.my_character && (
                  <Overlay
                    style={pct(result.my_character.character, result.image.width, result.image.height)}
                    color={COLORS.mine}
                    label="MINE"
                    bold
                  />
                )}
              </div>
            )}
          </div>

          {result && (
            <div className="grid grid-cols-3 gap-4">
              <Metric color={COLORS.char} n={result.characters.length} label="character" />
              <Metric color={COLORS.uid} n={result.user_ids.length} label="user_id" />
              <Metric
                color={COLORS.mine}
                n={result.my_character ? 1 : 0}
                label="내 캐릭터"
                text={result.my_character ? "선정됨" : "없음"}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Overlay({
  style,
  color,
  label,
  bold,
}: {
  style: React.CSSProperties;
  color: string;
  label: string;
  bold?: boolean;
}) {
  return (
    <div
      className="absolute pointer-events-none"
      style={{ ...style, border: `${bold ? 3 : 2}px solid ${color}`, boxShadow: bold ? `0 0 0 1px ${color}` : undefined }}
    >
      <span
        className="absolute -top-5 left-0 text-[11px] font-semibold px-1 rounded-sm whitespace-nowrap"
        style={{ background: color, color: "#06121a" }}
      >
        {label}
      </span>
    </div>
  );
}

function Metric({ color, n, label, text }: { color: string; n: number; label: string; text?: string }) {
  return (
    <div className="card p-4">
      <div className="text-2xl font-bold" style={{ color }}>
        {text ?? n}
      </div>
      <div className="text-xs text-muted mt-1">{label}</div>
    </div>
  );
}
