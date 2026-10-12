import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { anchorClaims, host } from "../lib/citations";
import type { Claim, EvidenceMap, Source } from "../lib/types";
import { Bar, Card } from "./ui";

const QUALITY_STYLE = {
  FULL: "border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-100",
  PARTIAL: "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100",
  DEGRADED: "border-red-400 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100",
} as const;

const QUALITY_TEXT = {
  FULL: "All model and tool calls used live providers.",
  PARTIAL: "Some steps used an offline fallback. Treat coverage as incomplete.",
  DEGRADED: "Report content or judgment came from offline mocks. Do not trust this report or its metrics.",
} as const;

export function Report({ report, map }: { report: string; map: EvidenceMap }) {
  const sources = useMemo(() => new Map(map.sources.map((s) => [s.id, s])), [map.sources]);
  const claimsById = useMemo(() => new Map(map.claims.map((c) => [c.id, c])), [map.claims]);
  const { markdown, unanchored } = useMemo(() => anchorClaims(report, map.claims), [report, map.claims]);
  const q = map.execution_quality;
  const er = map.evidence_report;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="space-y-4">
        <div role={q.status === "FULL" ? "status" : "alert"} className={`rounded-xl border p-3 text-sm ${QUALITY_STYLE[q.status]}`}>
          <strong>Execution quality: {q.status}.</strong> {QUALITY_TEXT[q.status]}
          {q.reason.length > 0 && <ul className="mt-1 list-disc pl-5">{q.reason.map((r) => <li key={r}>{r}</li>)}</ul>}
        </div>

        <Card>
          <article className="prose-report">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              urlTransform={(u) => u}
              components={{
                a: ({ href, children }) => {
                  const id = href?.startsWith("#cite-") ? href.slice(6) : null;
                  const claim = id ? claimsById.get(id) : undefined;
                  // Never render attacker-controlled hrefs: only our own citation anchors become elements.
                  return claim ? <CitationChip n={String(children)} claim={claim} sources={sources} confidence={er.claim_confidence?.[claim.id]} /> : <span>{children}</span>;
                },
              }}
            >
              {markdown}
            </ReactMarkdown>
          </article>
        </Card>

        {unanchored.length > 0 && (
          <Card title="Claims not located in the text">
            <ul className="space-y-2 text-sm">
              {unanchored.map((c) => (
                <li key={c.id}>
                  {c.text} <SourceList ids={c.source_ids} sources={sources} />
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>

      <aside className="space-y-4">
        {er.citation_completeness && (
          <Card title="Citation completeness">
            <Bar value={er.citation_completeness.completeness_score} />
            <p className="mt-1 text-xs text-stone-500">{Math.round(er.citation_completeness.completeness_score * 100)}% of claims cited</p>
            {er.citation_completeness.uncited_claims.length > 0 && (
              <ul className="mt-2 list-disc pl-4 text-xs text-red-700 dark:text-red-300">
                {er.citation_completeness.uncited_claims.map((c) => <li key={c}>Uncited: {c}</li>)}
              </ul>
            )}
          </Card>
        )}
        {er.coverage && (
          <Card title="Coverage by dimension">
            <div className="space-y-2">
              {Object.entries(er.coverage).map(([d, v]) => (
                <div key={d}>
                  <div className="flex justify-between text-xs"><span>{d}</span><span>{Math.round(v * 100)}%</span></div>
                  <Bar value={v} />
                </div>
              ))}
            </div>
          </Card>
        )}
        {er.contradictions && er.contradictions.length > 0 && (
          <Card title="Contradictions">
            <ul className="space-y-2 text-sm">
              {er.contradictions.map((c) => (
                <li key={c.topic}>
                  <div className="font-medium">{c.topic} <span className="text-xs text-amber-700 dark:text-amber-300">({c.status})</span></div>
                  <div className="text-xs text-stone-500">{c.reason}</div>
                  <SourceList ids={c.sources} sources={sources} />
                </li>
              ))}
            </ul>
          </Card>
        )}
        <Card title={`Sources (${map.sources.length})`}>
          <ul className="space-y-3 text-sm">
            {map.sources.map((s) => (
              <li key={s.id}>
                <div className="font-medium">{s.title}</div>
                <div className="text-xs text-stone-500">
                  {host(s.url)} · {er.source_types?.[s.id] ?? s.source_type ?? "unclassified"}
                </div>
              </li>
            ))}
          </ul>
          {Object.keys(map.rejection_reasons).length > 0 && (
            <details className="mt-3 text-xs">
              <summary className="cursor-pointer text-stone-500">{Object.keys(map.rejection_reasons).length} rejected</summary>
              <ul className="mt-1 list-disc pl-4">
                {Object.entries(map.rejection_reasons).map(([id, why]) => <li key={id}>{why}</li>)}
              </ul>
            </details>
          )}
        </Card>
      </aside>
    </div>
  );
}

function SourceList({ ids, sources }: { ids: string[]; sources: Map<string, Source> }) {
  if (ids.length === 0) return <span className="text-xs text-red-700 dark:text-red-300">(no source)</span>;
  return (
    <span className="text-xs text-stone-500">
      {ids.map((id) => sources.get(id)?.title ?? id).join(" · ")}
    </span>
  );
}

function CitationChip({ n, claim, sources, confidence }: { n: string; claim: Claim; sources: Map<string, Source>; confidence?: string }) {
  // Click / focus / hover all open the card, so touch and keyboard users are covered.
  const [open, setOpen] = useState(false);
  const uncited = claim.source_ids.length === 0;
  return (
    <span className="relative inline-block align-baseline" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        aria-expanded={open}
        aria-label={`Citation ${n}${uncited ? ", uncited claim" : ""}`}
        onClick={() => setOpen((o) => !o)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className={`mx-0.5 rounded px-1.5 text-[11px] font-semibold ${uncited ? "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200" : "bg-indigo-100 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-200"}`}
      >
        {uncited ? "!" : n}
      </button>
      {open && (
        <span role="tooltip" className="absolute left-0 top-6 z-10 block w-72 rounded-lg border border-stone-200 bg-white p-3 text-left text-xs shadow-lg dark:border-stone-700 dark:bg-stone-900">
          {confidence && <span className="mb-1 block font-semibold uppercase text-stone-500">confidence: {confidence}</span>}
          {uncited && <span className="block text-red-700 dark:text-red-300">No source was cited for this claim.</span>}
          {claim.source_ids.map((id) => {
            const s = sources.get(id);
            return (
              <span key={id} className="mb-2 block">
                <span className="block font-medium">{s?.title ?? id}</span>
                {s && <span className="block text-stone-500">{host(s.url)}</span>}
                {s?.snippet && <span className="mt-1 block italic text-stone-600 dark:text-stone-400">“{s.snippet}”</span>}
              </span>
            );
          })}
        </span>
      )}
    </span>
  );
}
