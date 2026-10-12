import { useState } from "react";
import type { BudgetMode } from "../lib/types";
import { Button, Card } from "./ui";

const MODES: { id: BudgetMode; hint: string }[] = [
  { id: "fast", hint: "Cheaper synthesis and verification; lowest latency" },
  { id: "balanced", hint: "Default tier assignment" },
  { id: "thorough", hint: "Stronger planner and validator; stricter source filtering" },
];

export function NewRun({ busy, onStart }: { busy: boolean; onStart: (topic: string, mode: BudgetMode) => void }) {
  const [topic, setTopic] = useState("");
  const [mode, setMode] = useState<BudgetMode>("balanced");
  return (
    <Card>
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (topic.trim()) onStart(topic.trim(), mode);
        }}
      >
        <label className="block">
          <span className="text-sm font-medium">Research topic</span>
          <textarea
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            rows={3}
            placeholder="e.g. Small language models for enterprise use"
            className="mt-1 w-full rounded-lg border border-stone-300 bg-transparent p-3 dark:border-stone-700"
          />
        </label>
        <fieldset className="grid gap-2 sm:grid-cols-3">
          <legend className="mb-1 text-sm font-medium">Budget mode</legend>
          {MODES.map((m) => (
            <label key={m.id} className={`cursor-pointer rounded-lg border p-3 text-sm ${mode === m.id ? "border-indigo-500 ring-1 ring-indigo-500" : "border-stone-300 dark:border-stone-700"}`}>
              <input type="radio" name="mode" className="sr-only" checked={mode === m.id} onChange={() => setMode(m.id)} />
              <div className="font-medium capitalize">{m.id}</div>
              <div className="text-xs text-stone-500">{m.hint}</div>
            </label>
          ))}
        </fieldset>
        <Button disabled={busy || !topic.trim()}>{busy ? "Starting… (planning)" : "Start research"}</Button>
      </form>
    </Card>
  );
}
