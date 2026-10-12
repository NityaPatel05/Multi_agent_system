import type { ButtonHTMLAttributes, ReactNode } from "react";

export const Card = ({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) => (
  <section className={`rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900 ${className}`}>
    {title && <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-stone-500">{title}</h3>}
    {children}
  </section>
);

export const Button = ({ variant = "primary", className = "", ...p }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" }) => (
  <button
    {...p}
    className={`rounded-lg px-4 py-2 text-sm font-medium transition disabled:opacity-50 ${
      variant === "primary"
        ? "bg-indigo-600 text-white hover:bg-indigo-500"
        : "border border-stone-300 hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800"
    } ${className}`}
  />
);

export const Bar = ({ value }: { value: number }) => (
  <div className="h-2 w-full rounded bg-stone-200 dark:bg-stone-800" role="meter" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
    <div className="h-2 rounded bg-indigo-500" style={{ width: `${Math.round(value * 100)}%` }} />
  </div>
);
