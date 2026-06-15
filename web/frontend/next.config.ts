import type { NextConfig } from "next";

// 백엔드(FastAPI) 원본. 기본은 로컬. 배포 시에도 프론트와 같은 머신에서 돌므로 동일.
const BACKEND = process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // /api·/static 을 백엔드로 프록시 → 프론트와 동일 출처(relative)로 호출.
  // Cloudflare Tunnel로 프론트(:3000) 하나만 공개해도 백엔드까지 함께 노출된다 (CORS 불필요).
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/static/:path*", destination: `${BACKEND}/static/:path*` },
    ];
  },
};

export default nextConfig;
