"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, TrackMetrics } from "@/lib/api";

type Status = "idle" | "uploading" | "running" | "done" | "error";

export default function TrackPage() {
  const [status, setStatus] = useState<Status>("idle");
  const [token, setToken] = useState<string | null>(null);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrackMetrics | null>(null);
  const [pace, setPace] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const upload = useCallback(
    async (file: File) => {
      setError(null);
      setMetrics(null);
      setStatus("uploading");
      try {
        const fd = new FormData();
        fd.append("video", file);
        const res = await fetch(`${API_BASE}/api/track/upload`, { method: "POST", body: fd });
        if (!res.ok) throw new Error("업로드 실패");
        const { token } = await res.json();
        setToken(token);
        setStatus("running");
        // MJPEG 스트림 연결 → 서버에서 추적 시작
        setStreamUrl(`${API_BASE}/api/track/stream/${token}?pace=${pace ? 1 : 0}`);
      } catch (e) {
        setError(e instanceof Error ? e.message : "오류");
        setStatus("error");
      }
    },
    [pace]
  );

  // 추적 중 지표 폴링
  useEffect(() => {
    if (status !== "running" || !token) return;
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/track/metrics/${token}`).then((x) => x.json());
        if (r.metrics) setMetrics(r.metrics);
        if (r.status === "done") {
          setStatus("done");
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        /* 무시 */
      }
    }, 1000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [status, token]);

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="mb-8">
        <p className="text-brand text-sm font-medium uppercase tracking-wide mb-2">영상 추적</p>
        <h1 className="text-3xl font-bold">동영상에서 내 캐릭터 추적</h1>
        <p className="text-muted mt-2">
          ByteTrack + hysteresis 락으로 내 캐릭터(<span className="text-mine">MINE</span>)에 락온하고,
          라벨 없이 측정 가능한 추적 지표를 실시간으로 산출합니다.
        </p>
      </header>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="space-y-5">
          <label className="card grid place-items-center text-center p-8 cursor-pointer hover:border-brand transition-colors">
            <input
              type="file"
              accept="video/*"
              className="hidden"
              disabled={status === "running" || status === "uploading"}
              onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
            />
            <div className="text-4xl mb-2">🎬</div>
            <p className="font-medium">영상을 클릭해 업로드</p>
            <p className="text-xs text-muted mt-1">MP4 · AVI · 짧은 클립 권장</p>
          </label>

          <label className="card p-4 flex items-center justify-between text-sm cursor-pointer">
            <span>실시간 속도로 재생 (pace)</span>
            <input
              type="checkbox"
              checked={pace}
              disabled={status === "running"}
              onChange={(e) => setPace(e.target.checked)}
              className="accent-[var(--brand)] w-4 h-4"
            />
          </label>

          <div className="card p-4 text-xs text-muted leading-relaxed">
            ⚠️ 긴 영상은 GPU 부하·처리 시간이 큽니다. 데모는 <span className="text-foreground">10~30초</span>
            클립을 권장합니다. 추적이 끝나면 주석 mp4를 내려받을 수 있습니다.
          </div>

          {status === "done" && token && (
            <a
              href={`${API_BASE}/api/track/download/${token}`}
              className="btn-brand block text-center px-5 py-3 text-sm"
            >
              주석 영상 다운로드 ↓
            </a>
          )}
        </div>

        <div className="lg:col-span-2 space-y-5">
          <div className="card p-3 min-h-[300px] grid place-items-center bg-black/40">
            {error && <p className="text-char text-sm">{error}</p>}
            {!streamUrl && !error && (
              <p className="text-muted text-sm">추적 영상이 여기에 스트리밍됩니다</p>
            )}
            {streamUrl && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={streamUrl} alt="tracking stream" className="w-full rounded-lg" />
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <M label="프레임" v={metrics?.frames ?? "—"} />
            <M label="추적률" v={metrics ? `${(metrics.my_located_rate * 100).toFixed(1)}%` : "—"} accent="mine" />
            <M label="캐릭터 탐지율" v={metrics ? `${(metrics.any_char_rate * 100).toFixed(1)}%` : "—"} />
            <M label="ID 스위치" v={metrics?.id_switches ?? "—"} />
          </div>
          {status === "running" && (
            <p className="text-sm text-brand animate-pulse">추적 중… (지표는 1초마다 갱신)</p>
          )}
        </div>
      </div>
    </div>
  );
}

function M({ label, v, accent }: { label: string; v: React.ReactNode; accent?: "mine" }) {
  return (
    <div className="card p-4">
      <div className={`text-2xl font-bold ${accent === "mine" ? "text-mine" : "text-foreground"}`}>{v}</div>
      <div className="text-xs text-muted mt-1">{label}</div>
    </div>
  );
}
