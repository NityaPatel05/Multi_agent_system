Read CLAUDE.md and docs/architecture.md's "Suggested next step" and "Build order" sections.

Identify the next unbuilt piece in this order:
1. state.py + main.py: Evidence Verifier + Critic nodes, Send() fan-out, conditional edges, SQLite checkpointer.
2. MCP adapter split (mcp_search_adapter.py, mcp_extract_adapter.py, mcp_orchestrator.py) + Groq/Gemini model router.
3. Observability (LangSmith tracing, judge_eval.py) + deployment (FastAPI api/main.py).

Check the current repo state (view the directory tree) to see what already exists, then implement
the smallest next increment only — don't jump ahead. Follow the hard rules in CLAUDE.md exactly
(Search agent never calls Exa/Tavily directly, all holistic judgment lives in Evidence Verifier
only, Critic never triggers a blanket re-run, concrete model names only in routing/model_tiers.py).

After implementing, list what was added and what the next increment should be.
