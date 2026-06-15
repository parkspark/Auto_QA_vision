// 백엔드 호출은 같은 출처의 상대경로(/api·/static)로 보내고, Next rewrites가 백엔드로 프록시한다.
// → Cloudflare Tunnel로 프론트만 공개해도 전체 동작하고 CORS가 필요 없다.
// 필요 시 NEXT_PUBLIC_API_BASE 로 절대 URL을 강제할 수 있다(예: 백엔드 별도 터널).
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export const api = (path: string) => `${API_BASE}${path}`;

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
}

export interface DetectResult {
  image: { width: number; height: number };
  characters: Box[];
  user_ids: Box[];
  my_character: { character: Box; user_id: Box } | null;
}

export interface TrackMetrics {
  frames: number;
  my_located_rate: number;
  any_char_rate: number;
  any_uid_rate: number;
  id_switches: number;
  max_coast_frames: number;
}

export async function detectImage(file: File, conf: number): Promise<DetectResult> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("conf", String(conf));
  const res = await fetch(api("/api/detect"), { method: "POST", body: fd });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? "탐지 실패");
  return res.json();
}

// 샘플은 서버에서 파일명으로 직접 탐지 (브라우저 cross-origin blob 문제 회피)
export async function detectSample(name: string, conf: number): Promise<DetectResult> {
  const res = await fetch(api(`/api/detect_sample?name=${encodeURIComponent(name)}&conf=${conf}`));
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? "탐지 실패");
  return res.json();
}
