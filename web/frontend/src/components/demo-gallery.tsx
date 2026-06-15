"use client";

import { useEffect, useState } from "react";
import { API_BASE, TrackMetrics } from "@/lib/api";

interface DemoVideo {
  name: string;
  url: string;
  size_mb: number;
}
interface DemoMetric extends Partial<TrackMetrics> {
  file: string;
  weights?: string;
  method?: string;
  video?: string;
}

// 대표 비교 한 쌍: v2 baseline vs v3 hysteresis
const FEATURED = ["track_baseline_v2_0725.mp4", "track_v3_hysteresis_0725.mp4"];

export function DemoGallery() {
  const [videos, setVideos] = useState<DemoVideo[]>([]);
  const [metrics, setMetrics] = useState<DemoMetric[]>([]);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/demos`)
      .then((r) => r.json())
      .then((d) => {
        setVideos(d.videos ?? []);
        setMetrics(d.metrics ?? []);
      })
      .catch(() => setErr(true));
  }, []);

  if (err)
    return (
      <div className="card p-6 text-muted text-sm">
        백엔드(API)에 연결할 수 없습니다. <code className="text-brand">start_web.bat</code> 으로 백엔드를 먼저 실행하세요.
      </div>
    );

  const featured = FEATURED.map((n) => videos.find((v) => v.name === n)).filter(Boolean) as DemoVideo[];
  const v2 = metrics.find((m) => m.weights?.includes("v2") && !m.method);
  const v3 = metrics.find((m) => m.weights?.includes("v3") && !m.method);

  return (
    <div className="space-y-8">
      <div className="grid md:grid-cols-2 gap-5">
        {featured.map((v, i) => {
          const isV3 = v.name.includes("v3");
          const m = isV3 ? v3 : v2;
          return (
            <div key={v.name} className="card overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-border">
                <span className="font-semibold">{isV3 ? "v3 + Hysteresis 락" : "v2 베이스라인"}</span>
                <span className={`chip ${isV3 ? "!text-mine !border-mine/40" : ""}`}>
                  ID 스위치 {m?.id_switches ?? "—"}
                </span>
              </div>
              {/* MJPEG/대용량 mp4 — 로컬 데모용 직접 재생 */}
              <video
                src={`${API_BASE}${v.url}`}
                controls
                preload="metadata"
                className="w-full aspect-video bg-black"
              />
              <div className="px-4 py-3 text-xs text-muted">
                {v.size_mb} MB · {m ? `탐지율 ${(m.any_char_rate! * 100).toFixed(1)}%` : ""}
              </div>
            </div>
          );
        })}
      </div>
      {v2 && v3 && (
        <p className="text-sm text-muted">
          동일 평가 영상(2025-07-25 holdout, {v3.frames} 프레임)에서 hysteresis 락이 ID 스위치를{" "}
          <span className="text-mine font-semibold">
            {v2.id_switches} → {v3.id_switches}
          </span>{" "}
          로 억제했습니다.
        </p>
      )}
    </div>
  );
}
