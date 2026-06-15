import { ReactNode } from "react";

export function Section({
  id,
  eyebrow,
  title,
  children,
  className = "",
}: {
  id?: string;
  eyebrow?: string;
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={`mx-auto max-w-6xl px-5 py-16 ${className}`}>
      {(eyebrow || title) && (
        <div className="mb-9">
          {eyebrow && (
            <p className="text-brand text-sm font-medium tracking-wide uppercase mb-2">{eyebrow}</p>
          )}
          {title && <h2 className="text-2xl sm:text-3xl font-bold">{title}</h2>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ value, label, sub }: { value: ReactNode; label: string; sub?: string }) {
  return (
    <div className="card p-5">
      <div className="text-2xl sm:text-3xl font-bold text-gradient">{value}</div>
      <div className="mt-1 text-sm font-medium">{label}</div>
      {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
    </div>
  );
}
