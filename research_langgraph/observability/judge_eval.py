"""LLM-as-Judge eval gate (observability layer, architecture.md §6).

Hard rule (CLAUDE.md): any tier-binding, model_router, or agent-prompt change must be run
against the saved topic set through this scorer and compared to the last known-good
baseline before merging — this is what `/eval-gate` runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from research_langgraph.routing.model_router import call_structured

_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
_BASELINE_PATH = _EVAL_DIR / "baseline_scores.json"
_TOPICS_DIR = _EVAL_DIR / "topics"

JUDGE_SYSTEM_PROMPT = """You are an impartial judge scoring a research report against its
source topic. Score three dimensions from 0.0 to 1.0:
- citation_accuracy: do claims in the report actually match what their cited sources say?
- coverage: does the report address the topic's expected dimensions thoroughly?
- hallucination_rate: groundedness, where 1.0 = no hallucinated/uncited claims found and
  0.0 = pervasive hallucination (higher is better, despite the name).
Return the three scores plus a one-sentence rationale for each.
"""


class JudgeScore(BaseModel):
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


def _mock_judge(topic: str, report: str, evidence_map: dict) -> JudgeScore:
    claims = evidence_map.get("claims", [])
    cited = [c for c in claims if c.get("source_ids")]
    citation_accuracy = round(len(cited) / max(1, len(claims)), 2)

    coverage_map = evidence_map.get("evidence_report", {}).get("coverage", {})
    coverage = round(sum(coverage_map.values()) / max(1, len(coverage_map)), 2) if coverage_map else 0.5

    completeness = evidence_map.get("evidence_report", {}).get("citation_completeness", {})
    hallucination_rate = completeness.get("completeness_score", citation_accuracy)

    return JudgeScore(
        citation_accuracy=citation_accuracy,
        coverage=coverage,
        hallucination_rate=hallucination_rate,
        rationale="Offline mock judge: heuristic score derived from the evidence map.",
    )


def judge_report(topic: str, report: str, evidence_map: dict, budget_mode: str = "balanced") -> JudgeScore:
    parsed, _result = call_structured(
        node_name="critic",  # reuses the Medium tier; the judge isn't a graph node itself
        system=JUDGE_SYSTEM_PROMPT,
        user=f"Topic: {topic}\nReport:\n{report}\nEvidence map: {evidence_map}",
        schema=JudgeScore,
        mock_fn=lambda: _mock_judge(topic, report, evidence_map),
        budget_mode=budget_mode,
    )
    return parsed


def load_baseline() -> dict:
    if _BASELINE_PATH.exists():
        content = _BASELINE_PATH.read_text().strip()
        if content:
            return json.loads(content)
    return {}


def save_baseline(scores: dict) -> None:
    _BASELINE_PATH.write_text(json.dumps(scores, indent=2))


def load_topics() -> list[str]:
    if not _TOPICS_DIR.exists():
        return []
    return [f.read_text().strip() for f in sorted(_TOPICS_DIR.glob("*.txt")) if f.read_text().strip()]


def run_eval_gate(topics: Optional[list[str]] = None) -> dict:
    """Run every saved topic through the graph + judge, compare against the baseline.
    Returns {"results": {...}, "regressions": [...], "baseline": {...}, "failures": [...]}.

    One topic crashing (run_research now re-raises after marking the run "failed" — see
    main.py's mark_run_failed) must not lose the scores already collected for every other
    topic, so each topic is isolated in its own try/except; failed topics are recorded
    under "failures" and excluded from the baseline/regression comparison rather than
    silently aborting the whole gate.
    """
    from research_langgraph.main import run_research

    topics = topics if topics is not None else load_topics()
    baseline = load_baseline()
    results: dict[str, dict] = {}
    regressions: list[dict] = []
    failures: list[dict] = []

    for topic in topics:
        try:
            final_state = run_research(topic, auto_approve=True)
            score = judge_report(
                topic, final_state.get("final_report", ""), final_state.get("evidence_map", {})
            )
        except Exception as e:
            failures.append({"topic": topic, "error": str(e)})
            continue

        results[topic] = score.model_dump()

        prior = baseline.get(topic)
        if prior:
            for dim in ("citation_accuracy", "coverage", "hallucination_rate"):
                if results[topic][dim] < prior.get(dim, 0):
                    regressions.append({
                        "topic": topic,
                        "dimension": dim,
                        "baseline": prior.get(dim),
                        "new": results[topic][dim],
                    })

    return {"results": results, "regressions": regressions, "baseline": baseline, "failures": failures}


if __name__ == "__main__":
    report = run_eval_gate()
    print(json.dumps(report, indent=2))
    if report["failures"]:
        print(f"\n{len(report['failures'])} topic(s) FAILED to run — do not ship until resolved.")
    elif report["regressions"]:
        print(f"\nREGRESSIONS FOUND ({len(report['regressions'])}) — do not ship.")
    else:
        save_baseline(report["results"])
        print("\nNo regressions. Baseline updated.")
