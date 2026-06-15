"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "개요" },
  { href: "/detect", label: "라이브 탐지" },
  { href: "/track", label: "영상 추적" },
];

export function SiteNav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-50 border-b border-border/80 backdrop-blur-md bg-background/70">
      <nav className="mx-auto max-w-6xl flex items-center justify-between px-5 h-16">
        <Link href="/" className="flex items-center gap-2.5 font-semibold">
          <span className="grid place-items-center w-8 h-8 rounded-lg bg-brand/15 text-brand text-lg">◉</span>
          <span>
            DNF<span className="text-brand">Vision</span>
          </span>
        </Link>
        <div className="flex items-center gap-1">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`px-3.5 py-2 rounded-lg text-sm transition-colors ${
                  active ? "text-brand bg-brand/10" : "text-muted hover:text-foreground"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
