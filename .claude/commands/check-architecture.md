Audit the current diff (or whole repo if no diff given) against the "Hard rules" section of
CLAUDE.md. Specifically check:

- Does search_agent.py call any HTTP/search-provider client directly instead of going through
  tools/mcp_search_adapter.py?
- Does anything outside evidence_verifier_agent.py compute coverage, contradictions,
  source-type/diversity, or citation completeness?
- Does critic_agent.py (or the graph's conditional edges) ever trigger a full restart instead of
  a targeted re-run?
- Does anything outside routing/model_tiers.py hardcode a concrete model name?
- Is there any vector-DB/RAG call in the live web-search path (as opposed to memory/episodic_store.py)?

Report violations as a checklist with file:line references. If clean, say so explicitly.
