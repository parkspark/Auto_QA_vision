// 버전별 val mAP50 막대 차트 — 외부 차트 라이브러리 없이 순수 SVG.
const DATA = [
  { v: "v1", map: 0.856, note: "실사 817 + 합성 2,000" },
  { v: "v2", map: 0.923, note: "1·2차 검수 반영" },
  { v: "v3", map: 0.931, note: "영상 프레임 721 추가 (현역)", current: true },
];

export function VersionChart() {
  const W = 560;
  const H = 240;
  const pad = { l: 44, r: 16, t: 16, b: 36 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const min = 0.8;
  const max = 0.95;
  const y = (v: number) => pad.t + innerH - ((v - min) / (max - min)) * innerH;
  const bw = innerW / DATA.length;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="버전별 mAP50 비교">
      {[0.8, 0.85, 0.9, 0.95].map((g) => (
        <g key={g}>
          <line x1={pad.l} x2={W - pad.r} y1={y(g)} y2={y(g)} stroke="var(--border)" strokeWidth={1} />
          <text x={pad.l - 8} y={y(g) + 4} textAnchor="end" fontSize={11} fill="var(--muted)">
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      {DATA.map((d, i) => {
        const x = pad.l + i * bw + bw * 0.2;
        const w = bw * 0.6;
        const top = y(d.map);
        return (
          <g key={d.v}>
            <rect
              x={x}
              y={top}
              width={w}
              height={pad.t + innerH - top}
              rx={6}
              fill={d.current ? "var(--brand)" : "var(--accent)"}
              opacity={d.current ? 1 : 0.55}
            />
            <text x={x + w / 2} y={top - 8} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--foreground)">
              {d.map.toFixed(3)}
            </text>
            <text x={x + w / 2} y={H - 14} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--foreground)">
              {d.v}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
