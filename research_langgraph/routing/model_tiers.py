# The only place concrete model names should appear. See CLAUDE.md hard rules.
# Re-bound 2026-10-09: qwen/qwen3.6-27b (old Medium) shut down 2026-09-14 and
# llama-3.3-70b-versatile is flagged for shutdown on Groq's deprecations page. Tiers
# stay distinct so budget_mode keeps working. Run observability/judge_eval.py before merging.
TIER_BINDINGS = {
    "Small":  {"provider": "groq",   "model": "openai/gpt-oss-20b"},
    "Medium": {"provider": "groq",   "model": "openai/gpt-oss-120b"},
    "High":   {"provider": "groq",   "model": "openai/gpt-oss-120b"},
}
FALLBACK_BINDINGS = {
    "Small":  {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    "Medium": {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    "High":   {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
}

# Episodic-memory embeddings (memory/episodic_store.py). Replaces text-embedding-004, which
# Google shut down 2026-01-14. Output dimension is set by the store (_EMBED_DIM).
EMBEDDING_BINDING = {"provider": "gemini", "model": "gemini-embedding-001"}

# LLM-as-judge (observability/judge_eval.py) deliberately runs on a DIFFERENT provider/model
# family than the pipeline that wrote the report (Groq/gpt-oss), so the model isn't grading
# its own output. Gemini first, Groq only as a last resort (self-grading is then flagged).
JUDGE_BINDINGS = {
    "primary":  {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    "fallback": {"provider": "groq",   "model": "openai/gpt-oss-120b"},
}
