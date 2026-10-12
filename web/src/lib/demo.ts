import type { Plan, RunSnapshot } from "./types";

// Opened with /?demo=1 — lets the UI be built and reviewed without live API keys or
// the (still changing) backend. Shapes match schemas/validated_outputs.py.
export const demoPlan: Plan = {
  dimensions: ["Definition", "Architecture", "Applications", "Limitations"],
  queries_by_dimension: {
    Definition: ["what is a small language model", "SLM vs LLM parameter count"],
    Architecture: ["distillation quantization small language models"],
    Applications: ["on-device SLM use cases", "enterprise SLM deployment"],
    Limitations: ["small language model limitations reasoning"],
  },
  complexity: "medium",
  rationale: "Broad survey topic with four natural facets; two search rounds and one targeted re-run.",
};

export const demoDone: RunSnapshot = {
  run_id: "demo",
  status: "done",
  awaiting_approval: false,
  plan: demoPlan,
  final_report: `# Small Language Models

## Definition

Small language models are language models with a few billion parameters or fewer. They trade some breadth for speed and cost.

## Architecture

Most SLMs are built by distilling a larger teacher model and then quantizing the weights. Quantization to 4 bits cuts memory use roughly fourfold.

## Applications

SLMs run on phones and laptops without a network connection. Enterprises use them for narrow, high-volume tasks.

## Limitations

SLMs show weaker multi-step reasoning than frontier models. Some vendors claim parity on every benchmark.

---
**⚠ Execution quality: PARTIAL.** One or more steps used an offline fallback.`,
  evidence_map: {
    claims: [
      { id: "c1", text: "Small language models are language models with a few billion parameters or fewer.", dimension: "Definition", source_ids: ["s1", "s2"] },
      { id: "c2", text: "Most SLMs are built by distilling a larger teacher model and then quantizing the weights.", dimension: "Architecture", source_ids: ["s2"] },
      { id: "c3", text: "Quantization to 4 bits cuts memory use roughly fourfold.", dimension: "Architecture", source_ids: ["s3"] },
      { id: "c4", text: "SLMs run on phones and laptops without a network connection.", dimension: "Applications", source_ids: ["s4"] },
      { id: "c5", text: "SLMs show weaker multi-step reasoning than frontier models.", dimension: "Limitations", source_ids: ["s1"] },
      { id: "c6", text: "Some vendors claim parity on every benchmark.", dimension: "Limitations", source_ids: [] },
    ],
    evidence_report: {
      claim_confidence: { c1: "high", c2: "high", c3: "medium", c4: "medium", c5: "high", c6: "low" },
      coverage: { Definition: 0.95, Architecture: 0.85, Applications: 0.6, Limitations: 0.7 },
      contradictions: [
        { topic: "Reasoning parity", sources: ["s1", "s4"], status: "Needs clarification", reason: "s4 claims parity on reasoning benchmarks; s1 measures a gap." },
      ],
      source_types: { s1: "academic", s2: "academic", s3: "blog", s4: "vendor" },
      source_diversity: { academic: 0.5, blog: 0.25, vendor: 0.25 },
      diversity_flag: "",
      citation_completeness: { uncited_claims: ["Some vendors claim parity on every benchmark."], completeness_score: 0.83 },
    },
    sources: [
      { id: "s1", url: "https://arxiv.org/abs/2410.00001", title: "A Survey of Small Language Models", snippet: "We survey models under 10B parameters and measure the reasoning gap against frontier models.", dimension: "Definition", provider: "exa", score: 0.92 },
      { id: "s2", url: "https://arxiv.org/abs/2410.00002", title: "Distilling Capability into Compact Models", snippet: "Distillation from a teacher followed by quantization is the dominant recipe.", dimension: "Architecture", provider: "exa", score: 0.88 },
      { id: "s3", url: "https://www.example-blog.com/quantization", title: "4-bit Quantization Explained", snippet: "Going from 16-bit to 4-bit weights reduces memory about 4x.", dimension: "Architecture", provider: "tavily", score: 0.7 },
      { id: "s4", url: "https://vendor.example.com/slm-launch", title: "Introducing our on-device model", snippet: "Runs fully offline on a laptop and matches larger models on reasoning.", dimension: "Applications", provider: "tavily", score: 0.55 },
    ],
    execution_quality: { status: "PARTIAL", reason: ["Critic used an offline fallback"] },
    fallback_history: [{ node: "critic", reason: "groq rate limited", fallback: "gemini" }],
    rejection_reasons: { s9: "low credibility: anonymous forum post", s10: "duplicate of s2" },
  },
};
