import { useCallback, useEffect, useRef, useState } from "react";
import { approvePlan, getRun, startRun, subscribe } from "./lib/api";
import { demoDone, demoPlan } from "./lib/demo";
import type { BudgetMode, RunSnapshot, StreamEvent } from "./lib/types";
import { Audit } from "./components/Audit";
import { LiveRun } from "./components/LiveRun";
import { NewRun } from "./components/NewRun";
import { PlanReview } from "./components/PlanReview";
import { Report } from "./components/Report";
import { Button, Card } from "./components/ui";

const DEMO = new URLSearchParams(location.search).get("demo");

export default function App() {
  const [run, setRun] = useState<RunSnapshot | null>(() =>
    DEMO === "plan" ? { run_id: "demo", status: "awaiting_approval", awaiting_approval: true, plan: demoPlan } : DEMO ? demoDone : null,
  );
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"report" | "audit">("report");
  const unsub = useRef<() => void>(() => {});

  const refresh = useCallback(async (id: string) => {
    const snap = await getRun(id);
    setRun(snap);
    return snap;
  }, []);

  // While a run is in flight, follow the SSE stream and re-read the snapshot on each change.
  useEffect(() => {
    if (DEMO || !run || run.status === "done" || run.status === "failed" || run.awaiting_approval) return;
    unsub.current();
    unsub.current = subscribe(
      run.run_id,
      (e) => {
        setEvents((prev) => [...prev, e]);
        refresh(run.run_id).catch((err) => setError(String(err.message ?? err)));
      },
      () => refresh(run.run_id).catch(() => {}),
    );
    return () => unsub.current();
  }, [run?.run_id, run?.status, run?.awaiting_approval, refresh]); // eslint-disable-line react-hooks/exhaustive-deps

  const guarded = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onStart = (topic: string, mode: BudgetMode) =>
    guarded(async () => {
      setEvents([]);
      const { run_id } = await startRun(topic, mode); // blocks until the approval gate today
      await refresh(run_id);
    });

  const onDecide = (approved: boolean, feedback: string) =>
    guarded(async () => {
      if (!run) return;
      await approvePlan(run.run_id, approved, feedback); // blocks until done or next interrupt today
      await refresh(run.run_id);
    });

  const reset = () => {
    unsub.current();
    setRun(null);
    setEvents([]);
    setError("");
    if (DEMO) history.replaceState(null, "", location.pathname);
  };

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Research Engine</h1>
        {run && <Button variant="ghost" onClick={reset}>New research</Button>}
      </header>

      {error && <div role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">{error}</div>}

      {!run && <NewRun busy={busy} onStart={onStart} />}

      {run?.awaiting_approval && <PlanReview plan={run.plan} busy={busy} onDecide={onDecide} />}

      {run && !run.awaiting_approval && run.status !== "done" && run.status !== "failed" && <LiveRun events={events} />}

      {run?.status === "failed" && (
        <Card title="Run failed">
          <p className="text-sm">The run stopped with an error. Start a new research to try again.</p>
        </Card>
      )}

      {run?.status === "done" && run.final_report && run.evidence_map && (
        <>
          <div role="tablist" className="flex gap-2">
            {(["report", "audit"] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                className={`rounded-lg px-3 py-1.5 text-sm capitalize ${tab === t ? "bg-indigo-600 text-white" : "hover:bg-stone-200 dark:hover:bg-stone-800"}`}>
                {t}
              </button>
            ))}
          </div>
          {tab === "report" ? <Report report={run.final_report} map={run.evidence_map} /> : <Audit map={run.evidence_map} />}
        </>
      )}
    </div>
  );
}
