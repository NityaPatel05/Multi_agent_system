import { useState } from "react";
import type { Plan } from "../lib/types";
import { Button, Card } from "./ui";

// Budget table from the README / main.py: complexity -> (search rounds, re-run budget)
const BUDGET = { simple: [1, 0], medium: [2, 1], complex: [2, 3] } as const;

interface Props {
  plan?: Plan;
  busy: boolean;
  onDecide: (approved: boolean, feedback: string) => void;
}

export function PlanReview({ plan, busy, onDecide }: Props) {
  const [feedback, setFeedback] = useState("");
  const actions = (
    <div className="mt-4 space-y-2">
      <textarea
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        rows={2}
        placeholder="Optional: what should change? (sent to the planner if you request changes)"
        className="w-full rounded-lg border border-stone-300 bg-transparent p-2 text-sm dark:border-stone-700"
      />
      <div className="flex gap-2">
        <Button disabled={busy} onClick={() => onDecide(true, "")}>
          {busy ? "Working…" : "Approve & run"}
        </Button>
        <Button variant="ghost" disabled={busy || !feedback.trim()} onClick={() => onDecide(false, feedback)}>
          Request changes
        </Button>
      </div>
    </div>
  );

  if (!plan)
    return (
      <Card title="Plan review">
        <p className="text-sm">
          The run is paused at the approval gate, but no plan came back from the API. You can still approve or request changes below.
        </p>
        {actions}
      </Card>
    );

  const [rounds, reruns] = BUDGET[plan.complexity];
  return (
    <Card title="Plan review — approve before any search budget is spent">
      <p className="mb-3 text-sm text-stone-600 dark:text-stone-400">{plan.rationale}</p>
      <div className="mb-4 flex flex-wrap gap-2 text-xs">
        <span className="rounded bg-indigo-100 px-2 py-1 font-medium text-indigo-800 dark:bg-indigo-950 dark:text-indigo-200">
          complexity: {plan.complexity}
        </span>
        <span className="rounded bg-stone-100 px-2 py-1 dark:bg-stone-800">{rounds} search round(s)</span>
        <span className="rounded bg-stone-100 px-2 py-1 dark:bg-stone-800">{reruns} targeted re-run(s)</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {plan.dimensions.map((d) => (
          <div key={d} className="rounded-lg border border-stone-200 p-3 dark:border-stone-800">
            <div className="font-medium">{d}</div>
            <ul className="mt-1 list-disc pl-5 text-sm text-stone-600 dark:text-stone-400">
              {(plan.queries_by_dimension[d] ?? []).map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      {actions}
    </Card>
  );
}
