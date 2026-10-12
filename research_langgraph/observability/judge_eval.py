"""LLM-as-Judge eval gate (observability layer, architecture.md §6).

Hard rule (CLAUDE.md): any tier-binding, model_router, or agent-prompt change must be run
against the saved topic set through this scorer and compared to the last known-good
baseline before merging — this is what `/eval-gate` runs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from research_langgraph.routing.model_router import call_structured
from research_langgraph.routing.model_tiers import JUDGE_BINDINGS

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
_BASELINE_PATH = _EVAL_DIR / "baseline_scores.json"
_TOPICS_DIR = _EVAL_DIR / "topics"

JUDGE_SYSTEM_PROMPT = """You are an impartial judge scoring a research report against its
source topic. Score three dimensions from 0.0 to 1.0. ALL are higher-is-better. Judge each
independently and be strict; most reports should NOT score above 0.8. Anchors:
- citation_accuracy: 1.0 = every claim matches what its cited source says; 0.7 = most do,
  a few are loosely supported; 0.4 = many claims are only tangentially supported; 0.0 =
  citations do not support the claims.
- coverage: 1.0 = every expected dimension has a dedicated, substantive, specific section;
  0.7 = all present but some thin; 0.4 = one or more dimensions missing or superficial;
  0.0 = the report ignores the topic's main aspects.
- hallucination_rate (groundedness; higher is better despite the name): 1.0 = no
  unsupported or invented facts/figures found; 0.7 = a few unsupported details; 0.4 =
  several unsupported claims; 0.0 = pervasive invention.
Return the three scores plus a one-sentence rationale for each.
"""


class JudgeScore(BaseModel):
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


def _cap_with_grounding(score: JudgeScore, evidence_map: dict) -> JudgeScore:
    """The LLM judge is never allowed to be MORE generous than the deterministic checks."""
    g = (evidence_map.get("evidence_report") or {}).get("grounding") or {}
    if not g:
        return score
    return score.model_copy(update={
        "citation_accuracy": min(score.citation_accuracy, g.get("claim_support_rate", 1.0)),
        "hallucination_rate": min(score.hallucination_rate, g.get("cited_claim_rate", 1.0)),
    })


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
    return judge_report_checked(topic, report, evidence_map, budget_mode)[0]


def judge_report_checked(topic: str, report: str, evidence_map: dict,
                         budget_mode: str = "balanced") -> tuple[JudgeScore, bool]:
    """Returns (score, used_mock) so the gate can refuse to trust offline/mock scores."""
    parsed, _result = call_structured(
        node_name="critic",  # reuses the Medium tier; the judge isn't a graph node itself
        system=JUDGE_SYSTEM_PROMPT,
        user=f"Topic: {topic}\nReport:\n{report}\nEvidence map: {evidence_map}",
        schema=JudgeScore,
        mock_fn=lambda: _mock_judge(topic, report, evidence_map),
        budget_mode=budget_mode,
        bindings=(JUDGE_BINDINGS["primary"], JUDGE_BINDINGS["fallback"]),
    )
    if _result.provider == JUDGE_BINDINGS["fallback"]["provider"] and not _result.used_mock:
        logger.warning("judge fell back to the pipeline's own provider; scores are self-graded")
    parsed = _cap_with_grounding(parsed, evidence_map)
    return parsed, _result.used_mock


EVAL_TOLERANCE = float(os.environ.get("EVAL_TOLERANCE", "0.05"))


def _worse(dim: str, new: float, old: float) -> bool:
    # all three scores are higher-is-better (hallucination_rate is groundedness)
    return new < old - EVAL_TOLERANCE


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
            score, judge_mock = judge_report_checked(
                topic, final_state.get("final_report", ""), final_state.get("evidence_map", {})
            )
            quality = (final_state.get("execution_quality") or {}).get("status")
            if judge_mock or quality == "DEGRADED":
                raise RuntimeError(
                    f"untrustworthy run (judge_mock={judge_mock}, execution_quality={quality}); "
                    "scores from mocks must not be compared to or saved as a baseline"
                )
        except Exception as e:
            failures.append({"topic": topic, "error": str(e)})
            continue

        results[topic] = score.model_dump()

        prior = baseline.get(topic)
        if prior:
            for dim in ("citation_accuracy", "coverage", "hallucination_rate"):
                # tolerance: an LLM judge is noisy, so only flag drops beyond it
                if _worse(dim, results[topic][dim], prior.get(dim, 0)):
                    regressions.append({
                        "topic": topic,
                        "dimension": dim,
                        "baseline": prior.get(dim),
                        "new": results[topic][dim],
                    })

    return {"results": results, "regressions": regressions, "baseline": baseline, "failures": failures}


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from research_langgraph.memory.checkpointer import close_checkpointers

    load_dotenv()  # entrypoint-only
    report = run_eval_gate()
    close_checkpointers()
    print(json.dumps(report, indent=2))
    if report["failures"]:
        print(f"\n{len(report['failures'])} topic(s) FAILED to run - do not ship until resolved.")
        sys.exit(1)
    if report["regressions"]:
        print(f"\nREGRESSIONS FOUND ({len(report['regressions'])}) - do not ship.")
        sys.exit(1)
    if "--update-baseline" in sys.argv:
        save_baseline(report["results"])
        print("\nNo regressions. Baseline updated.")
    else:
        print("\nNo regressions. (Pass --update-baseline to save these scores as the new baseline.)")
