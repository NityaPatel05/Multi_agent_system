# The only place concrete model names should appear. See CLAUDE.md hard rules.
TIER_BINDINGS = {
    "Small":  {"provider": "groq",   "model": "openai/gpt-oss-20b"},
    "Medium": {"provider": "groq",   "model": "qwen/qwen3.6-27b"},
    "High":   {"provider": "groq",   "model": "openai/gpt-oss-120b"},
}
FALLBACK_BINDINGS = {
    # gemini-2.5-flash-lite was retired ("no longer available to new users") — found via
    # live testing 2026-07-18; gemini-2.5-flash confirmed working in the same test run.
    "Small":  {"provider": "gemini", "model": "gemini-2.5-flash"},
    "Medium": {"provider": "gemini", "model": "gemini-2.5-flash"},
    "High":   {"provider": "gemini", "model": "gemini-2.5-flash"},
}
