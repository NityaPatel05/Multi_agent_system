from research_langgraph.agents import grounding
from research_langgraph.agents.evidence_verifier_agent import _apply_grounding
from research_langgraph.agents.synthesizer_agent import synthesizer_agent
from research_langgraph.observability.judge_eval import JudgeScore, _cap_with_grounding, _worse
from research_langgraph.schemas.validated_outputs import EvidenceVerifierOutput

DOC_A = {"source_id": "a", "dimension": "Architecture",
         "text": "The transformer architecture uses self-attention layers. GPT-3 has 175 billion parameters."}


def test_invalid_citations_are_dropped():
    claims = [{"id": "c1", "text": "x", "source_ids": ["a", "ghost"]}]
    cleaned, removed = grounding.drop_invalid_citations(claims, [DOC_A])
    assert cleaned[0]["source_ids"] == ["a"] and removed == 1


def test_support_detects_fabricated_numbers_and_unrelated_citations():
    ok = grounding.claim_support("GPT-3 has 175 billion parameters.", [DOC_A])
    fake_number = grounding.claim_support("GPT-3 has 999 billion parameters.", [DOC_A])
    unrelated = grounding.claim_support("Quantum annealing hardware reduces cryogenic overhead.", [DOC_A])
    assert ok >= 0.8 and fake_number < ok - 0.3 and unrelated < 0.3
    assert grounding.claim_support("anything at all", []) == 0.0


def test_coverage_needs_a_section_and_two_supported_claims():
    claims = [
        {"id": "1", "text": "GPT-3 has 175 billion parameters.", "dimension": "Architecture", "source_ids": ["a"]},
        {"id": "2", "text": "The transformer architecture uses self-attention layers.", "dimension": "Architecture", "source_ids": ["a"]},
        {"id": "3", "text": "One lonely claim about training data volumes", "dimension": "Training", "source_ids": ["a"]},
    ]
    evals = grounding.evaluate_claims(claims, [DOC_A])
    report = "## Architecture\n\ntext\n\n## Training\n\ntext"
    cov = grounding.dimension_coverage(["Architecture", "Training", "Recent Research"], claims, evals, report)
    assert cov["Architecture"] == 1.0
    assert cov["Training"] < 0.5           # cited but unsupported by the source text
    assert cov["Recent Research"] == 0.0   # no section, no claims


def test_verifier_takes_the_stricter_score_over_an_inflated_llm_score():
    llm = EvidenceVerifierOutput(coverage={"Architecture": 1.0, "Recent Research": 1.0})
    state = {"report_markdown": "## Architecture\n\ntext", "citations_removed": 0}
    claims = [{"id": "1", "text": "GPT-3 has 175 billion parameters.", "dimension": "Architecture", "source_ids": ["a"]}]
    out = _apply_grounding(llm, state, ["Architecture", "Recent Research"], claims, [DOC_A])
    assert out["coverage"]["Recent Research"] == 0.0     # LLM said 1.0, evidence says no
    assert out["coverage"]["Architecture"] == 0.5        # only one supported claim
    assert out["grounding"]["claim_support_rate"] == 1.0


def test_judge_cannot_exceed_deterministic_metrics():
    em = {"evidence_report": {"grounding": {"claim_support_rate": 0.4, "cited_claim_rate": 0.6}}}
    capped = _cap_with_grounding(JudgeScore(citation_accuracy=0.95, coverage=0.9, hallucination_rate=0.95), em)
    assert capped.citation_accuracy == 0.4 and capped.hallucination_rate == 0.6 and capped.coverage == 0.9


def test_all_judge_scores_are_higher_is_better():
    assert _worse("hallucination_rate", 0.5, 0.9) is True
    assert _worse("hallucination_rate", 0.95, 0.9) is False


def test_synthesizer_writes_one_section_per_dimension_in_order():
    docs = [{"source_id": f"s{i}", "dimension": d, "dimensions": [d], "text": f"{d} details " * 20}
            for i, d in enumerate(["Alpha", "Beta", "Gamma"])]
    out = synthesizer_agent({"topic": "T", "documents": docs, "dimensions": ["Alpha", "Beta", "Gamma", "Delta"]})
    r = out["report_markdown"]
    heads = [ln for ln in r.splitlines() if ln.startswith("## ")]
    assert heads == ["## Alpha", "## Beta", "## Gamma", "## Delta"]
    assert "No sources were retrieved" in r.split("## Delta")[1]
    assert {c["dimension"] for c in out["claims"]} == {"Alpha", "Beta", "Gamma"}
