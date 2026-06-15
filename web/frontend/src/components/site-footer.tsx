export function SiteFooter() {
  return (
    <footer className="border-t border-border/80 mt-24">
      <div className="mx-auto max-w-6xl px-5 py-10 flex flex-col sm:flex-row items-center justify-between gap-3 text-sm text-muted">
        <p>
          <span className="text-foreground font-medium">DNF Vision</span> — YOLO11s · ByteTrack ·
          FastAPI · Next.js
        </p>
        <p>던전앤파이터 캐릭터 탐지 &amp; 추적 · 포트폴리오 데모</p>
      </div>
    </footer>
  );
}
