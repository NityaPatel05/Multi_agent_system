import type { StreamEvent } from "../lib/types";
import { Card } from "./ui";

// Fixed order of the graph (README "Graph Flow"). Parallel stages are marked.
const STAGES = [
  ["planner", "Planner"],
  ["human_approval", "Approval"],
  ["search_agent", "Search ×N dimensions"],
  ["validator", "Validator"],
  ["extractor_agent", "Extract ×M sources"],
  ["synthesizer", "Synthesizer"],
  ["evidence_verifier", "Evidence Verifier"],
  ["critic", "Critic"],
  ["finalize", "Finalize"],
] as const;

export function LiveRun({ events }: { events: StreamEvent[] }) {
  const seen = new Map<string, "start" | "end">();
  events.forEach((e) => e.node && seen.set(e.node, e.phase ?? "start"));
  return (
    <Card title="Run in progress">
      {seen.size === 0 && (
        <p className="mb-3 text-sm text-stone-500">
          Waiting for the server. The stream currently reports only coarse status; per-step progress appears here once the
          API emits node events.
        </p>
      )}
      <ol className="space-y-2">
        {STAGES.map(([id, label]) => {
          const s = seen.get(id);
          const dot = s === "end" ? "bg-emerald-500" : s === "start" ? "animate-pulse bg-indigo-500" : "bg-stone-300 dark:bg-stone-700";
          return (
            <li key={id} className="flex items-center gap-3 text-sm">
              <span className={`h-2.5 w-2.5 rounded-full ${dot}`} />
              <span className={s ? "font-medium" : "text-stone-500"}>{label}</span>
            </li>
          );
        })}
      </ol>
    </Card>
  );
}
