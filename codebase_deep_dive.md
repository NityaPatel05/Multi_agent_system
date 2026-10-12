# 🔬 Multi-Agent Research System — Complete Codebase Deep Dive

## What Is This Project?

You give it a **topic** (e.g. `"small language models"`). It automatically:
1. Breaks that topic into research dimensions
2. Searches the web across all dimensions in parallel
3. Extracts and reads the actual page content
4. Writes a structured markdown research report
5. Verifies every claim against its cited sources
6. Critiques the report and triggers targeted re-runs if quality is low
7. Returns a final report + a structured evidence map

The whole pipeline is a **LangGraph state machine** — a directed graph of nodes (agents) connected by conditional edges.

---

## 📁 Directory Structure Overview

```
research_langgraph_setup/
├── research_langgraph/         ← the main Python package
│   ├── state.py                ← shared memory: the "whiteboard" all agents write to
│   ├── main.py                 ← the graph wiring + all conditional routing logic
│   ├── agents/                 ← one file per agent node
│   │   ├── planner_agent.py
│   │   ├── search_agent.py
│   │   ├── search_merge.py
│   │   ├── validator_agent.py
│   │   ├── extractor_agent.py
│   │   ├── extraction_merge.py
│   │   ├── synthesizer_agent.py
│   │   ├── evidence_verifier_agent.py
│   │   └── critic_agent.py
│   ├── routing/                ← model selection + LLM calling
│   │   ├── model_tiers.py      ← concrete model names (ONLY place)
│   │   └── model_router.py     ← call_structured() — the universal LLM caller
│   ├── tools/                  ← search + extraction adapters
│   │   ├── mcp_orchestrator.py ← single call surface for all tool calls
│   │   ├── mcp_search_adapter.py   ← Exa → Tavily → Mock search
│   │   ├── mcp_extract_adapter.py  ← HTML → PDF → raw → Mock extract
│   │   ├── mcp_search_server.py    ← optional Docker microservice (search)
│   │   └── mcp_extract_server.py   ← optional Docker microservice (extract)
│   ├── schemas/
│   │   └── validated_outputs.py  ← Pydantic contracts between every agent
│   ├── memory/
│   │   ├── checkpointer.py     ← Postgres or in-memory checkpoint (resume on crash)
│   │   └── episodic_store.py   ← vector store of past completed runs
│   ├── observability/
│   │   ├── execution_quality.py  ← FULL / PARTIAL / DEGRADED quality signal
│   │   ├── judge_eval.py         ← LangSmith evaluator
│   │   └── langsmith_setup.py    ← LangSmith tracing config
│   └── api/
│       └── main.py             ← FastAPI HTTP layer
├── schemas/validated_outputs.py ← Pydantic output contracts
├── .env / .env.example         ← API keys + config
└── requirements.txt
```

---

## 🧠 The Shared Whiteboard: `state.py`

**File:** [`state.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/state.py)

This is the most important concept. The entire system shares **one big Python TypedDict** called `ResearchState`. Think of it as a whiteboard that every agent can read and write.

LangGraph passes this state object from node to node. Each agent:
- **Reads** the fields it needs from state
- **Writes back** only the fields it produces

### Key fields and who owns them:

| Field | Written By | Read By |
|---|---|---|
| `topic` | You (the caller) | Planner, Synthesizer |
| `dimensions` | Planner | Supervisor, Extractor, Evidence Verifier |
| `queries_by_dimension` | Planner | Search fan-out routing |
| `complexity` | Planner | Supervisor (to set budget) |
| `raw_search_results` | Search Agent (each branch) | Search Merge |
| `sources` | Search Merge | Validator |
| `validated_sources` | Validator | Extractor fan-out |
| `extracted_docs` | Extractor Agent (each branch) | Extraction Merge |
| `documents` | Extraction Merge | Synthesizer |
| `report_markdown` | Synthesizer | Evidence Verifier, Critic, Finalize |
| `claims` | Synthesizer | Evidence Verifier |
| `evidence_report` | Evidence Verifier | Critic |
| `approved` | Critic | Routing (go to finalize or rerun) |
| `rerun_directives` | Critic | Prepare Rerun node |
| `final_report` | Finalize | You (the caller) |
| `audit_log` | Every agent | Observability |

### Special: List-append fields (parallel-safe)
```python
raw_search_results: Annotated[List[dict], operator.add]
extracted_docs:     Annotated[List[dict], operator.add]
audit_log:          Annotated[List[AuditEntry], operator.add]
token_usage:        Annotated[List[TokenUsage], operator.add]
```
When multiple agents write to these fields **at the same time** (parallel `Send()` fan-out), LangGraph **concatenates** the lists instead of overwriting. Every other field is last-write-wins.

---

## 🕸️ The Graph: `main.py`

**File:** [`main.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/main.py)

This file does **three things only**: wires nodes together, defines routing logic, and assembles the graph. No business logic lives here.

### The Full Flow Diagram

```
START
  │
  ▼
[planner]  ──────────────────────────────────────────────────────────────────────────────┐
  │                                                                                       │ (broaden: skip approval)
  │ (first time)                                                                          │
  ▼                                                                                       │
[human_approval] ←──── user sees plan, approves or requests changes                     │
  │                                                                                       │
  │ approved? ──► [supervisor_init_budget] sets search_rounds + rerun_budget             │
  │ rejected? ──► back to [planner] with revision hint                                   │
  │                                                                                       │
  ▼                                                                                       │
[search_agent] × N  (one per dimension, running in PARALLEL via Send())                  │
  │                                                                                       │
  ▼                                                                                       │
[search_merge]  ← dedup + normalize all parallel branches' results                       │
  │                                                                                       │
  ▼                                                                                       │
[validator]  ← LLM credibility check                                                     │
  │                                                                                       │
  ├── too few credible sources? ──► [consume_search_round] ──► back to [planner] ────────┘
  │
  ▼
[extractor_agent] × M  (one per validated URL, running in PARALLEL via Send())
  │
  ▼
[extraction_merge]  ← drop failed extractions, log fallbacks
  │
  ▼
[synthesizer]  ← LLM writes the markdown report + lists claims
  │
  ▼
[evidence_verifier]  ← LLM does all holistic checks (coverage, contradictions, diversity, citation)
  │
  ▼
[critic]  ← LLM reviews evidence report, either approves or issues targeted directives
  │
  ├── approved? ──► [finalize]
  ├── budget exhausted? ──► [finalize] (best effort)
  │
  └── not approved + budget left ──► [prepare_rerun]
        │
        ├── needs new search? ──► [search_agent] × K  (targeted re-search only those dimensions)
        │                           └──► [extraction_merge] ──► [synthesizer] ──► [evidence_verifier] ──► [critic]
        │
        └── only uncited claims? ──► [synthesizer]  (skip search entirely, just revise)
              └──► [evidence_verifier] ──► [critic]

[finalize]
  │
  ▼
END
```

### Budget Table (hardcoded in `main.py`)
```python
_ROUND_BUDGET = {
    "simple":  {"search_rounds_allowed": 1, "rerun_budget": 0},
    "medium":  {"search_rounds_allowed": 2, "rerun_budget": 1},
    "complex": {"search_rounds_allowed": 2, "rerun_budget": 3},
}
```
The Planner says `complexity = "complex"` → Supervisor gives 2 search rounds and 3 re-run budget. This is set **once** in code, not left to the model.

---

## 👾 Agent-by-Agent Deep Dive

---

### 1. 🗺️ Planner Agent
**File:** [`agents/planner_agent.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/planner_agent.py)

**What it does:**  
Given a topic like `"small language models"`, the Planner asks an LLM to break it into research dimensions and generate search queries for each.

**Input from state:**
- `topic` — the research topic string
- `budget_mode` — fast / balanced / thorough
- `is_broaden` (optional) — True if validator found too few sources last round
- `plan_revision_hint` (optional) — if human rejected the plan with feedback

**What the LLM is asked:**
> "Decompose the topic into DIMENSIONS a complete answer needs (Definition, Architecture, Applications, Limitations, Recent Research). For each dimension, generate 2–4 concrete search queries. Estimate complexity."

**Output written to state:**
```python
{
  "dimensions": ["Definition", "Architecture", "Applications", "Limitations"],
  "queries_by_dimension": {
      "Definition": ["what are small language models", "SLM definition overview"],
      "Architecture": ["SLM architecture transformer", "small model design patterns"],
      ...
  },
  "complexity": "medium",         # simple | medium | complex
  "plan_rationale": "...",
  "status": "awaiting_approval",  # pauses here for human review
  "audit_log": [...],
  "token_usage": [...]
}
```

**Validated against:** `PlannerOutput` Pydantic schema — must have at least 1 dimension, every dimension must have queries.

---

### 2. ✋ Human Approval Node (inline in `main.py`)
**File:** [`main.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/main.py#L46-L63)

This is NOT a separate file — it's a regular Python function in `main.py` that calls `interrupt()`.

**What it does:**  
Calls LangGraph's `interrupt()` to PAUSE the run. The run freezes here until a human (or the API's auto-approve) sends a response.

The human sees:
```json
{
  "type": "plan_approval",
  "dimensions": ["Definition", "Architecture", ...],
  "queries_by_dimension": {...},
  "complexity": "medium",
  "rationale": "..."
}
```

They respond with `{"approved": true}` or `{"approved": false, "feedback": "add more on training data"}`.

**Output:**
```python
{"plan_approved": True/False, "plan_revision_hint": "...", "status": "running"}
```

If `approved=False` → the graph loops back to Planner with the feedback as a hint.

---

### 3. 🏦 Supervisor Init Budget (inline in `main.py`)
**File:** [`main.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/main.py#L66-L79)

**What it does:**  
Reads the Planner's `complexity` and sets the hardcoded budget.

**Output:**
```python
{
  "search_rounds_allowed": 2,  # how many times validator can trigger a new search round
  "search_rounds_used": 1,     # counts the current first round as "1"
  "rerun_budget": 1,           # how many critic → rerun loops are allowed
  "rerun_used": 0,
  "is_broaden": False,
  "status": "running"
}
```

---

### 4. 🔍 Search Agent (runs N times in PARALLEL)
**File:** [`agents/search_agent.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/search_agent.py)

**How it's invoked:**  
`route_to_search()` in `main.py` creates a `Send()` call for **each dimension** in `queries_by_dimension`. If there are 4 dimensions, 4 Search Agent instances run **concurrently**.

```python
# Each Send() call creates an isolated copy of state with just these fields:
Send("search_agent", {"dimension": "Architecture", "queries": ["SLM architecture ...", ...]})
```

**What it does:**  
Routes to the MCP orchestrator → search adapter → Exa or Tavily. Stamps each result with the dimension it belongs to.

**Output (appended to `raw_search_results`):**
```python
# Each branch appends its sources to the shared list (operator.add reducer)
{"raw_search_results": [
    {"id": "src_abc123", "url": "https://...", "title": "...", "snippet": "...",
     "dimension": "Architecture", "provider": "exa", "score": 0.85},
    ...
]}
```

**Never calls Exa/Tavily directly** — always goes through `mcp_orchestrator.dispatch_search()`.

---

### 5. 🔀 Search Merge
**File:** [`agents/search_merge.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/search_merge.py)

**What it does:**  
Runs AFTER all parallel Search Agent branches finish. Takes the combined `raw_search_results` list and:
1. Deduplicates by URL (keeps highest-scored copy)
2. Normalizes field shapes

**Input:**  
`raw_search_results` — all sources from all parallel branches combined

**Output written to state:**
```python
{"sources": [
    {
        "id": "src_abc123",
        "url": "https://example.com/paper",
        "title": "Small LMs Survey",
        "snippet": "...",
        "dimension": "Architecture",
        "provider": "exa",
        "score": 0.85,
        "source_type": None   # ← intentionally None here; only Evidence Verifier classifies this
    },
    ...
]}
```

**Does NOT**: classify source type or score diversity — that's the Evidence Verifier's exclusive job.

---

### 6. ✅ Validator Agent
**File:** [`agents/validator_agent.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/validator_agent.py)

**What it does:**  
Asks an LLM to do a basic **credibility gate** — reject spam, broken URLs, off-topic results. Only returns `accepted_ids` and `rejected_ids` — never full source objects.

**Input from state:**  `sources` (from Search Merge)

**What the LLM is asked:**  
Given a brief list of {id, url, title, snippet} for each source, classify each into accepted or rejected with a reason.

**Why only IDs?** Because if the LLM had to return the full source objects, it might silently change a field like `dimension`. By only returning IDs, the validator agent reconstructs full records in Python from the original list — the model never touches the data directly.

**Output written to state:**
```python
{
    "validated_sources": [  # full source dicts, reconstructed in Python, not by the LLM
        {"id": "src_abc123", "url": "...", "dimension": "Architecture", ...},
        ...
    ],
    "rejected_urls": ["https://spam-site.com/..."],
    "credible_count": 8,           # number of accepted sources
    "needs_broader_search": False, # True if credible_count < 3
    "rejection_reasons": {
        "src_xyz789": "Blog source with no citations"
    }
}
```

**Routing after Validator:**
- `credible_count < 3` AND rounds not exhausted → `consume_search_round` → back to Planner (broaden search)
- Otherwise → fan-out to Extractor Agent

---

### 7. 📥 Extractor Agent (runs M times in PARALLEL)
**File:** [`agents/extractor_agent.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/extractor_agent.py)

**How it's invoked:**  
`route_to_extract()` creates a `Send()` for each `validated_source` that hasn't been extracted yet (avoids re-extracting on reruns).

**What it does:**  
Downloads the actual page content for one URL. Calls `mcp_orchestrator.dispatch_extract()` → MCP extract adapter.

**Output (appended to `extracted_docs`):**
```python
{"extracted_docs": [{
    "source_id": "src_abc123",
    "url": "https://...",
    "dimension": "Architecture",
    "text": "Full page text content here ...",
    "char_count": 12500,
    "extractor": "html",   # html | pdf | raw | mock
    "fell_back": False,    # True if primary parser failed but raw worked
    "failed": False,       # True if everything failed
    "error": ""
}]}
```

---

### 8. 🔀 Extraction Merge
**File:** [`agents/extraction_merge.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/extraction_merge.py)

**What it does:**  
Combines all parallel extraction branches, drops failures, and surfaces fallbacks.

**Output:**
```python
{
    "documents": [  # sorted by dimension, failed ones removed
        {"source_id": "...", "text": "...", "dimension": "Architecture", ...},
        ...
    ],
    "extraction_fallbacks": ["src_xyz789"],  # used raw-text instead of HTML
    "extraction_dropped": ["src_qqq111"]     # completely failed, dropped
}
```

---

### 9. ✍️ Synthesizer Agent
**File:** [`agents/synthesizer_agent.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/synthesizer_agent.py)

**What it does:**  
Asks an LLM to write a structured markdown research report from the extracted documents. The LLM also lists each factual claim it makes and which source IDs back it.

**Input from state:**  `topic`, `documents`

**What the LLM is asked:**
> "Write a well-organized markdown report covering every dimension. List the individual factual claims you made along with the source_ids backing each one. If a claim isn't backed by any document, still include it but leave source_ids empty."

**Output:**
```python
{
    "report_markdown": "# Small Language Models\n\n## Definition\n...\n## Architecture\n...",
    "claims": [
        {"id": "c0", "text": "SLMs typically have < 7B parameters",
         "dimension": "Definition", "source_ids": ["src_abc123", "src_def456"]},
        {"id": "c1", "text": "SLMs use transformer architecture",
         "dimension": "Architecture", "source_ids": []},  # uncited!
        ...
    ],
    "synthesizer_used_mock": False
}
```

**On targeted rerun (recite mode):**  
If the Critic flagged uncited claims, the Synthesizer is called again with `recite_targets` — a list of specific claim texts to fix. It gets the existing report and is told to only add citations for those claims or remove them.

---

### 10. 🔎 Evidence Verifier Agent
**File:** [`agents/evidence_verifier_agent.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/evidence_verifier_agent.py)

**What it does:**  
This is the **single holistic judgment stage**. Does ALL quality checks in one pass. Nothing else in the pipeline is allowed to do these checks.

**Input from state:**  `dimensions`, `claims`, `sources`, `documents`

**5 checks it does:**

| Check | What it produces |
|---|---|
| **Claim confidence** | Per-claim: High / Medium / Unverified |
| **Coverage** | Per-dimension score 0–1 (how well was that dimension researched?) |
| **Contradictions** | Sub-topics where sources disagree (with status: Resolved / Needs clarification) |
| **Source types** | Classify EVERY source: Academic / Government / Company / Documentation / GitHub / News / Books / Other |
| **Citation completeness** | List every claim with no supporting source_ids; compute completeness score |

**Output:**
```python
{
    "evidence_report": {
        "claim_confidence": {"c0": "High", "c1": "Unverified", ...},
        "coverage": {"Definition": 0.9, "Architecture": 0.6, "Limitations": 0.3},
        "contradictions": [
            {"topic": "SLM parameter count", "sources": ["src_abc"], "status": "Needs clarification"}
        ],
        "source_types": {"src_abc123": "Academic", "src_def456": "Company", ...},
        "source_diversity": {"Academic": 0.25, "Company": 0.50, "News": 0.25, ...},
        "diversity_flag": "poor — over-indexed on Company, little Academic/Government representation",
        "citation_completeness": {
            "uncited_claims": ["SLMs use transformer architecture"],
            "completeness_score": 0.80
        }
    },
    "evidence_verifier_used_mock": False
}
```

---

### 11. 🧑‍⚖️ Critic Agent
**File:** [`agents/critic_agent.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/agents/critic_agent.py)

**What it does:**  
Reads the Evidence Verifier's output and either **approves** the report or issues **targeted directives** to fix specific problems.

**Special: DEGRADED check first.**  
Before any LLM call, the Critic runs `compute_execution_quality()`. If `status == "DEGRADED"` (meaning the Synthesizer or Evidence Verifier fell back to mock data), the Critic **short-circuits with no LLM call** — a mock judgment on mock content proves nothing.

**Input from state:**  `evidence_report`, `report_markdown`

**Output when approved:**
```python
{
    "approved": True,
    "execution_status": "APPROVED",
    "retry_recommended": False,
    "rerun_directives": [],
    "critic_summary": "Approved.",
    "rerun_dimensions": [],
    "rerun_subtopics": [],
    "rerun_recite": False,
    "execution_quality": {"status": "FULL", "reason": []}
}
```

**Output when rejected:**
```python
{
    "approved": False,
    "execution_status": "REJECTED",
    "retry_recommended": True,
    "rerun_directives": [
        {"action": "research_dimension", "target": "Limitations", "reason": "coverage 0.3 below threshold"},
        {"action": "recite", "target": "SLMs use transformer architecture", "reason": "uncited claim"},
        {"action": "improve_diversity", "target": "prioritize Academic/Government", "reason": "poor diversity"}
    ],
    "critic_summary": "3 targeted issues found.",
    "rerun_dimensions": ["Limitations"],
    "rerun_subtopics": [],
    "rerun_recite": True
}
```

**Available directive actions:**
| Action | What triggers it | What happens |
|---|---|---|
| `research_dimension` | Coverage score < 0.7 for a dimension | Re-search + re-extract that dimension only |
| `resolve_contradiction` | An unresolved contradiction | Search for that sub-topic specifically |
| `improve_diversity` | Poor source type diversity | Re-search with source-type hint |
| `recite` | Uncited claims | Skip search, send back to Synthesizer only |
| `retry_degraded` | DEGRADED execution quality | Skip search, redo Synthesizer + Evidence Verifier |

---

### 12. 🔄 Prepare Rerun Node (inline in `main.py`)
**File:** [`main.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/main.py#L92-L135)

**What it does:**  
Translates the Critic's abstract directives into a concrete, **narrowed** search set.

Example: if Critic says `research_dimension: "Limitations"`, this node sets:
```python
active_queries_by_dimension = {"Limitations": ["<topic> Limitations", "<topic> Limitations overview"]}
```

If all directives are `recite` (fix uncited claims), it sets `rerun_search_needed = False` so the graph skips search and goes straight to Synthesizer.

---

### 13. 🏁 Finalize Node (inline in `main.py`)
**File:** [`main.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/main.py#L138-L169)

**What it does:**  
Assembles the final report. Appends a quality caveat if execution was PARTIAL or DEGRADED.

**Output:**
```python
{
    "final_report": "# Small Language Models\n\n## Definition\n...\n\n---\n*Note: some pipeline stages used fallback...*",
    "evidence_map": {
        "claims": [...],
        "evidence_report": {...},
        "sources": [...],
        "execution_quality": {"status": "PARTIAL", "reason": ["Mock Search"]},
        "fallback_history": [...],
        "rejection_reasons": {...}
    },
    "status": "done"
}
```

---

## ⚙️ The Model Router: `routing/model_router.py`

**File:** [`routing/model_router.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/routing/model_router.py)

This is the **universal LLM caller** used by every agent except Search and Extractor (which don't call LLMs).

### Tier Assignment

Each node is assigned an abstract tier — **not a concrete model name**:

```python
NODE_TIERS = {
    "planner":           "Medium",
    "search_agent":      "Small",   # no LLM call, but registered
    "validator":         "Small",
    "extractor":         "Small",
    "synthesizer":       "High",
    "evidence_verifier": "High",
    "critic":            "Medium",
}
```

Tiers shift based on `budget_mode`:
- `"fast"` → downgrades Synthesizer and Evidence Verifier by one tier
- `"thorough"` → upgrades Planner and Validator by one tier

### Concrete Models: `routing/model_tiers.py`

```python
TIER_BINDINGS = {
    "Small":  {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
    "Medium": {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
    "High":   {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
}
FALLBACK_BINDINGS = {
    "Small":  {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    "Medium": {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    "High":   {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
}
```

This is the **only file** where concrete model names appear.

### `call_structured()` — the core function

Every LLM-calling agent does:
```python
parsed, result = call_structured(
    node_name="planner",
    system=SYSTEM_PROMPT,
    user=f"Topic: {topic}",
    schema=PlannerOutput,           # Pydantic schema for validation
    mock_fn=lambda: _mock_plan(topic),  # offline fallback
    budget_mode=budget_mode,
)
```

Internally it does:
1. Resolve tier from node name + budget_mode
2. Try **Groq** (primary)
3. If Groq fails → try **Gemini** (fallback)
4. Parse the JSON response against the Pydantic schema
5. If parsing fails → one corrective retry against the same provider
6. If both providers fail → call `mock_fn()` (deterministic offline)
7. Return `(parsed_output, RoutedCallResult)` — the result object carries audit/token info

**The schema hint** is automatically appended to the system prompt:
```
Respond with ONLY a single JSON object matching this schema (no markdown fences, no commentary):
{"title": "PlannerOutput", "properties": {...}}
```

---

## 🔧 Tools Layer: `tools/`

### `mcp_orchestrator.py`
**File:** [`tools/mcp_orchestrator.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/tools/mcp_orchestrator.py)

The **single call surface** for all tool calls. Agents call `mcp_orchestrator.dispatch_search()` or `mcp_orchestrator.dispatch_extract()` — never the adapters directly.

Does two things only:
1. If `MCP_SEARCH_URL` / `MCP_EXTRACT_URL` env vars are set → POST to the Docker microservice
2. If those aren't set (local dev) → call the adapter in-process
3. If the Docker server is unreachable → fall back to in-process anyway

### `mcp_search_adapter.py`
**File:** [`tools/mcp_search_adapter.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/tools/mcp_search_adapter.py)

Fallback chain for search:
```
Exa (primary)
  ↓ fails?
Tavily (fallback)
  ↓ fails?
Mock (deterministic offline — deterministic by query hash so tests are stable)
```

Each result gets a deterministic ID: `src_ + SHA1(url)[:10]`

### `mcp_extract_adapter.py`
**File:** [`tools/mcp_extract_adapter.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/tools/mcp_extract_adapter.py)

Fallback chain for extraction:
```
HTML/PDF parser (BeautifulSoup / PyMuPDF)
  ↓ fails?
Raw text fallback (strip HTML tags, keep content)
  ↓ fails?
Mock (offline placeholder text)
  ↓ fails?
failed=True (source dropped)
```

Also has a security check: rejects non-http(s) URLs (prevents file:// SSRF).

---

## 📐 Schemas: `schemas/validated_outputs.py`

**File:** [`schemas/validated_outputs.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/schemas/validated_outputs.py)

Every agent's LLM output is validated against a Pydantic model here. This is the **contract between nodes**.

Key schemas:

| Schema | Used By |
|---|---|
| `PlannerOutput` | `planner_agent` |
| `ValidatorOutput` | `validator_agent` |
| `SynthesizerOutput` | `synthesizer_agent` |
| `EvidenceVerifierOutput` | `evidence_verifier_agent` |
| `CriticOutput` | `critic_agent` |

`CriticOutput` has a validator that enforces: "approved=True must have no directives" and "approved=False must have at least one directive."

---

## 💾 Memory: `memory/`

### `checkpointer.py`
**File:** [`memory/checkpointer.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/memory/checkpointer.py)

LangGraph checkpoints the full `ResearchState` after every node completes.

- If `DATABASE_URL` is set → uses **PostgresSaver** (production)
- Otherwise → **InMemorySaver** (local dev — doesn't survive restarts)

This means: if a node crashes mid-run, when you resume the graph, it picks up from the last successfully completed node.

### `episodic_store.py`
**File:** [`memory/episodic_store.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/memory/episodic_store.py)

A vector store of past completed runs, indexed by topic embedding.

- `store_run(...)` → saves a completed run
- `retrieve_similar_runs(topic)` → finds past runs on similar topics

Not wired into the live pipeline yet (designed for future use to seed new runs with already-validated sources).

Backends:
- `QDRANT_URL` set → uses **Qdrant** (vector DB)
- Otherwise → local JSON file with hash-based "embedding" (works without any infrastructure)

---

## 👁️ Observability: `observability/`

### `execution_quality.py`
**File:** [`observability/execution_quality.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/observability/execution_quality.py)

Classifies every run into 3 quality levels:

| Status | Meaning |
|---|---|
| `FULL` | Every agent used a real LLM + real search/extract provider |
| `PARTIAL` | Some non-critical agents (search, validator) used mock fallback |
| `DEGRADED` | Synthesizer or Evidence Verifier used mock — the report content itself or its evaluation is synthetic |

`DEGRADED` → Critic auto-rejects without calling any LLM (no point evaluating fake content).

---

## 🌐 API: `api/main.py`

**File:** [`api/main.py`](file:///c:/Users/patel/Desktop/multi_agent_real/research_langgraph_setup/research_langgraph/api/main.py)

FastAPI layer exposing 4 endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/research` | POST | Start a new research run; blocks until the plan-approval interrupt; returns `{run_id, status}` |
| `/research/{run_id}` | GET | Poll status; returns `{status, final_report, evidence_map, awaiting_approval}` |
| `/research/{run_id}/stream` | GET | SSE stream that pushes status updates in real-time |
| `/research/{run_id}/approve-plan` | POST | Resume the run after human approval; blocks until done |

---

## 🌊 Complete Data Flow Summary (End-to-End)

```
YOU call run_research("small language models")
│
├─► state: {topic: "small language models", run_id: "uuid", budget_mode: "balanced"}
│
[planner_agent] calls call_structured(schema=PlannerOutput) via Groq/Gemini
│  state ← {dimensions: [...], queries_by_dimension: {...}, complexity: "medium"}
│
[human_approval] interrupts — YOU approve the plan
│  state ← {plan_approved: True}
│
[supervisor_init_budget] sets budget
│  state ← {search_rounds_allowed: 2, rerun_budget: 1}
│
PARALLEL — for each dimension:
  [search_agent("Definition")] → mcp_orchestrator → mcp_search_adapter → Exa/Tavily/Mock
  [search_agent("Architecture")] → ...
  [search_agent("Applications")] → ...
  [search_agent("Limitations")] → ...
│  state ← raw_search_results: [all sources combined via operator.add]
│
[search_merge] dedup by URL, normalize
│  state ← sources: [normalized, deduplicated list]
│
[validator_agent] calls LLM → accept/reject each source
│  state ← {validated_sources: [...], credible_count: 8, needs_broader_search: False}
│
PARALLEL — for each validated source:
  [extractor_agent("src_abc123", "https://...")] → mcp_orchestrator → mcp_extract_adapter → HTML/PDF/raw/Mock
  [extractor_agent("src_def456", "https://...")] → ...
  ...
│  state ← extracted_docs: [all docs combined via operator.add]
│
[extraction_merge] drop failures, sort
│  state ← documents: [kept docs, sorted by dimension]
│
[synthesizer_agent] calls LLM → writes markdown report + claims list
│  state ← {report_markdown: "# Small Language Models\n...", claims: [...]}
│
[evidence_verifier_agent] calls LLM → 5-check holistic judgment
│  state ← evidence_report: {coverage: {...}, contradictions: [...], citation_completeness: {...}, ...}
│
[critic_agent]
│  → checks execution_quality first (is anything DEGRADED?)
│  → calls LLM → reads evidence_report → issues directives or approves
│  state ← {approved: False, rerun_directives: [{action: "research_dimension", target: "Limitations"}]}
│
[prepare_rerun] translates directives → narrowed query set
│  state ← active_queries_by_dimension: {"Limitations": ["SLM limitations", ...]}
│
PARALLEL — targeted re-search:
  [search_agent("Limitations")] → Exa/Tavily/Mock
│  state ← raw_search_results: [new sources appended]
│
[extraction_merge] → [synthesizer] → [evidence_verifier] → [critic]
│  (Critic approves this time)
│  state ← {approved: True}
│
[finalize]
│  state ← {
│    final_report: "# Small Language Models\n\n## Definition\n...",
│    evidence_map: {claims: [...], sources: [...], execution_quality: {...}},
│    status: "done"
│  }
│
DONE — final_report + evidence_map returned to you
```

---

## 🧩 Key Design Rules (Hard Rules Enforced Everywhere)

1. **Model names in ONE place only** — `model_tiers.py`. Every agent uses abstract tier strings.
2. **Source-type classification in ONE place only** — `evidence_verifier_agent.py`. No other node classifies sources.
3. **Holistic judgment in ONE place only** — Evidence Verifier. No other node computes coverage, contradictions, or diversity.
4. **Tool calls through ONE gateway** — always `mcp_orchestrator`, never adapters directly.
5. **Critic never blanket-restarts** — always issues targeted directives naming specific dimensions/claims.
6. **Budget set ONCE in code** — `_ROUND_BUDGET` table in `main.py`, derived from Planner's complexity.
7. **DEGRADED = automatic rejection** — if Synthesizer or Evidence Verifier used mock, Critic short-circuits with no LLM call.
8. **Every fallback is logged** — `audit_log`, `fallback_history`, `token_usage` in state capture every call.
