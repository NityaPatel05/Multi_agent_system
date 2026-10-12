import type { BudgetMode, RunSnapshot, StreamEvent } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

const post = (url: string, body: unknown) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// NOTE: today this call blocks until the graph reaches the approval interrupt.
export const startRun = (topic: string, budget_mode: BudgetMode) =>
  post("/research", { topic, budget_mode }).then(json<{ run_id: string; status: string }>);

// The API always sends a `plan` object (fields are null before the Planner runs) and names the
// rationale `plan_rationale`; normalise to our Plan shape, or undefined when there is no plan yet.
function normalise(raw: any): RunSnapshot {
  const p = raw.plan;
  const plan =
    p && Array.isArray(p.dimensions)
      ? { dimensions: p.dimensions, queries_by_dimension: p.queries_by_dimension ?? {}, complexity: p.complexity ?? "medium", rationale: p.plan_rationale ?? p.rationale }
      : undefined;
  return { ...raw, plan };
}

export const getRun = (id: string) => fetch(`/research/${id}`).then(json<any>).then(normalise);

export const approvePlan = (id: string, approved: boolean, feedback = "") =>
  post(`/research/${id}/approve-plan`, { approved, feedback }).then(json<{ run_id: string; status: string }>);

/** Subscribes to the SSE stream; returns an unsubscribe function. */
export function subscribe(id: string, onEvent: (e: StreamEvent) => void, onClose: () => void) {
  const es = new EventSource(`/research/${id}/stream`);
  es.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data));
    } catch {
      /* ignore malformed frame */
    }
  };
  es.onerror = () => {
    es.close();
    onClose();
  };
  return () => es.close();
}
