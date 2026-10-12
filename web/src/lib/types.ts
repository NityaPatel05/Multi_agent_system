// Mirrors research_langgraph/state.py + schemas/validated_outputs.py.
// Every field the backend is still changing is optional so the UI degrades, not breaks.

export type BudgetMode = "fast" | "balanced" | "thorough";
export type RunStatus = "running" | "awaiting_approval" | "done" | "failed" | "unknown";
export type QualityStatus = "FULL" | "PARTIAL" | "DEGRADED";

export interface Plan {
  dimensions: string[];
  queries_by_dimension: Record<string, string[]>;
  complexity: "simple" | "medium" | "complex";
  rationale?: string;
}

export interface Source {
  id: string;
  url: string;
  title: string;
  snippet?: string;
  dimension?: string;
  provider?: string;
  score?: number;
  source_type?: string | null;
}

export interface Claim {
  id: string;
  text: string;
  dimension?: string;
  source_ids: string[];
}

export interface Contradiction {
  topic: string;
  sources: string[];
  status: string;
  reason: string;
}

export interface EvidenceReport {
  claim_confidence?: Record<string, string>;
  coverage?: Record<string, number>;
  contradictions?: Contradiction[];
  source_types?: Record<string, string>;
  source_diversity?: Record<string, number>;
  diversity_flag?: string;
  citation_completeness?: { uncited_claims: string[]; completeness_score: number };
}

export interface FallbackEvent {
  node: string;
  reason: string;
  fallback: string;
}

export interface EvidenceMap {
  claims: Claim[];
  evidence_report: EvidenceReport;
  sources: Source[];
  execution_quality: { status: QualityStatus; reason: string[] };
  fallback_history: FallbackEvent[];
  rejection_reasons: Record<string, string>;
}

export interface RunSnapshot {
  run_id: string;
  status: RunStatus;
  awaiting_approval: boolean;
  plan?: Plan; // normalised from the API's `plan` object in lib/api.ts
  final_report?: string | null;
  evidence_map?: EvidenceMap | null;
}

export interface StreamEvent {
  run_id: string;
  status?: RunStatus;
  node?: string; // future: per-node events
  phase?: "start" | "end";
  detail?: string;
}
