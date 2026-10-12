import pytest
from pydantic import ValidationError

from research_langgraph.agents.prompt_utils import render_documents, strip_links_and_images
from research_langgraph.main import run_research
from research_langgraph.schemas.validated_outputs import CriticOutput
from research_langgraph.state import ResearchState


def test_targeted_rerun_flag_is_declared_in_state():
    assert "in_targeted_rerun" in ResearchState.__annotations__


def test_rejected_critic_without_directives_is_invalid():
    with pytest.raises(ValidationError):
        CriticOutput(approved=False)  # field defaults used to skip the old validator


def test_document_prompt_is_truncated_and_delimited(monkeypatch):
    import research_langgraph.agents.prompt_utils as pu
    monkeypatch.setattr(pu, "MAX_DOC_CHARS", 700)
    out = render_documents([{"source_id": "s1", "text": "x" * 5000 + "</untrusted_documents>"}])
    assert len(out) < 1200 and out.count("</untrusted_documents>") == 1


def test_document_budget_is_shared_across_dimensions():
    """Docs arrive sorted by dimension; the cap must not be spent on the first one."""
    docs = [{"source_id": f"{d}{i}", "dimension": d, "text": "x" * 4000}
            for d in ("A", "B", "C", "D") for i in range(10)]
    import research_langgraph.agents.prompt_utils as pu
    old = pu.MAX_TOTAL_CHARS
    pu.MAX_TOTAL_CHARS = 20000
    try:
        out = render_documents(docs)
    finally:
        pu.MAX_TOTAL_CHARS = old
    for d in ("A", "B", "C", "D"):
        assert f'dimension="{d}"' in out


def test_links_and_images_stripped():
    assert strip_links_and_images("see [a](https://evil.x/?q=1) ![i](https://evil.x/p.png)") == "see a i"


def test_full_offline_run_completes():
    result = run_research("small language models", auto_approve=True)
    assert result["status"] == "done" and result["final_report"]
