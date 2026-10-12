# Multi-Agent Research System

A production-grade, multi-agent research pipeline built on LangGraph. The system decomposes a research topic into structured dimensions, fans out parallel web searches, extracts and validates sources, synthesizes a cited report, verifies evidence, and runs a self-correcting quality loop — all orchestrated as a typed state graph with human-in-the-loop plan approval.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Graph Flow](#graph-flow)
  - [Agent Roles](#agent-roles)
  - [Model Routing Layer](#model-routing-layer)
  - [MCP Tool Layer](#mcp-tool-layer)
  - [Memory and Checkpointing](#memory-and-checkpointing)
  - [Observability](#observability)
  - [API Layer](#api-layer)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Docker Deployment](#docker-deployment)
- [Budget Modes](#budget-modes)
- [Execution Quality](#execution-quality)
- [Self-Healing Behavior](#self-healing-behavior)

---

## Overview

Given a research topic, the system:

1. Decomposes it into thematic dimensions (e.g. Definition, Architecture, Applications, Limitations)
2. Generates targeted search queries per dimension
3. Pauses for human approval of the research plan before spending any search budget
4. Fans out parallel searches using the `Send()` primitive across all dimensions simultaneously
5. Validates and deduplicates sources by credibility
6. Fans out parallel full-text extraction per validated source
7. Synthesizes a structured, cited markdown report with a claims list
8. Verifies each claim against extracted evidence
9. Runs a critic-driven quality loop that can trigger targeted re-searches or synthesis revisions
10. Produces a final report alongside a structured evidence map

The graph is backed by a Postgres checkpointer (LangGraph's persistent memory), exposing full state at every node transition for resumability and inspection.

---

## Architecture

### Graph Flow

The pipeline is assembled as a LangGraph `StateGraph` with a single shared `ResearchState`. The complete node execution order is:

```
START
  |
  v
Planner  <--------------------------------------------+
  |                                                   |
  | (first run: human approval gate)                  |
  v                                                   |
Human Approval                                        |
  |                                                   |
  | (approved)                                        |
  v                                                   |
Supervisor Init Budget                                |
  |                                                   |
  | Send() fan-out per dimension                      |
  v                                                   |
Search Agent [x N dimensions in parallel]             |
  |                                                   |
  v                                                   |
Search Merge  (reducer: list concatenation)           |
  |                                                   |
  v                                                   |
Validator                                             |
  |                                                   |
  | (too few credible sources + rounds remaining) ----+  (Consume Search Round -> back to Planner)
  |
  | Send() fan-out per validated source
  v
Extractor Agent [x M sources in parallel]
  |
  v
Extraction Merge  (reducer: list concatenation)
  |
  v
Synthesizer
  |
  v
Evidence Verifier
  |
  v
Critic
  |
  | (approved or budget exhausted)
  v
Finalize  --> END
  |
  | (not approved + budget remaining)
  v
Prepare Rerun
  |
  | (targeted re-search needed)
  +----> Search Agent [narrowed dimensions only] --> ... --> Critic
  |
  | (only synthesis revision needed)
  +----> Synthesizer --> ... --> Critic
```

The key design principle is that every fan-out uses LangGraph's `Send()` primitive, which dispatches branch-local state copies to the same node in parallel. The `raw_search_results` and `extracted_docs` fields use `operator.add` reducers so branch outputs are concatenated rather than overwritten.

### Agent Roles

| Agent | Model Tier | Responsibility |
|---|---|---|
| Planner | Medium | Decomposes topic into dimensions, generates queries, estimates complexity (simple / medium / complex) |
| Supervisor | Inline node | Reads complexity once, sets the round and re-run budget for the entire run |
| Human Approval | Interrupt | Pauses the graph and exposes the plan to a human before any search budget is spent |
| Search Agent | Small | Calls MCP search tools (Exa or Tavily) for a single dimension's queries |
| Search Merge | Inline node | Deduplicates and normalizes raw results from all parallel branches into a unified source list |
| Validator | Small | Scores each source for credibility, filters to a validated set, flags if broader search is needed |
| Extractor Agent | Small | Fetches full text for one source (PyMuPDF for PDFs, BeautifulSoup for HTML, raw fallback) |
| Extraction Merge | Inline node | Consolidates extracted documents from all parallel branches |
| Synthesizer | High | Produces a structured markdown report and claim list from extracted documents |
| Evidence Verifier | High | Performs holistic judgment on each claim against extracted evidence spans |
| Critic | Medium | Reviews the evidence report, approves or issues targeted directives (re-search a dimension, resolve a contradiction, recite an uncited claim, retry a degraded run) |
| Finalize | Inline node | Assembles the final report and structured evidence map, annotates with execution quality warnings |

### Model Routing Layer

All LLM calls are routed through a single entry point: `call_structured()` in `routing/model_router.py`. No agent file contains a concrete model name.

**Tier bindings** (defined only in `routing/model_tiers.py`):

| Tier | Primary (Groq) | Fallback (Gemini) |
|---|---|---|
| Small | openai/gpt-oss-20b | gemini-3.1-flash-lite |
| Medium | openai/gpt-oss-120b | gemini-3.1-flash-lite |
| High | openai/gpt-oss-120b | gemini-3.1-flash-lite |

**Fallback chain for every call:**

1. Resolve node name to abstract tier
2. Try primary provider (Groq) with structured JSON output
3. On validation error: retry once against the same provider with a corrective prompt
4. On provider failure or repeated validation failure: fall over to Gemini
5. If both providers are unavailable: execute the agent's deterministic `mock_fn` so the graph remains runnable offline

All fallover events are recorded in `ResearchState.fallback_history` and `audit_log`.

**Budget modes** shift tier assignments at call time without touching agent code:

| Mode | Effect |
|---|---|
| fast | Downgrades Synthesizer and Evidence Verifier by one tier |
| balanced | Default; no overrides |
| thorough | Upgrades Planner and Validator by one tier |

Concurrent `Send()` fan-outs to Small-tier nodes are bounded by per-provider semaphores (`GROQ_MAX_CONCURRENCY`, `GEMINI_MAX_CONCURRENCY`, both defaulting to 4) to stay within free-tier rate limits.

### MCP Tool Layer

Search and extraction are exposed as small HTTP tool services (REST, not the MCP wire protocol, despite the `mcp_*` file names) that can run in-process or as independent Docker containers.

**Search tools** (`tools/mcp_search_adapter.py`):
- Primary: Exa semantic search (requires `EXA_API_KEY`)
- Fallback: Tavily search (requires `TAVILY_API_KEY`)
- Final fallback: offline mock results

**Extraction tools** (`tools/mcp_extract_adapter.py`):
- Primary: PyMuPDF for PDF content
- Fallback: BeautifulSoup HTML extraction
- Final fallback: raw text fetch
- Terminal fallback: offline mock document

The `mcp_orchestrator.py` layer is intentionally thin: it passes requests to the adapters (in-process or over HTTP) and produces a uniform audit entry. No routing logic lives in the orchestrator.

When `MCP_SEARCH_URL` or `MCP_EXTRACT_URL` are set in the environment, the orchestrator dispatches over HTTP to the standalone server containers defined in `docker/Dockerfile.mcp`. If those containers are unreachable, it automatically falls back to running the adapter in-process.

### Memory and Checkpointing

State is persisted using LangGraph's Postgres checkpointer (`langgraph-checkpoint-postgres`). Every node transition writes to the checkpoint, so:

- A run interrupted at the human-approval gate can be resumed by POST-ing to `/research/{id}/approve-plan`
- A crashed run shows `status: failed` to polling clients rather than appearing stuck at `running`
- `mark_run_failed()` is called in every exception handler to ensure the terminal state is written

An in-memory checkpointer (not persistent across restarts) is used when `DATABASE_URL` is not set, keeping the system runnable for development without a running database.

The episodic memory store (`memory/episodic_store.py`) uses Qdrant to persist summaries of past runs, intended to let the planner learn from previous sessions. **Not wired into the graph yet** - `store_run` / `retrieve_similar_runs` are never called.

### Observability

The system integrates with LangSmith for distributed tracing. Every model call, tool call, and recovery event is appended to `ResearchState.audit_log` and `token_usage`, making the full execution trace available both through LangSmith and by reading the checkpoint directly.

The `observability/execution_quality.py` module tracks whether a run was executed with full provider access (`FULL`), partially degraded (`PARTIAL`), or fully mocked (`DEGRADED`). The Critic reads this status and can issue a `retry_degraded` directive when a DEGRADED run completes, prompting a fresh synthesis pass rather than a new search round.

The `observability/judge_eval.py` module provides an LLM-as-judge evaluator for offline quality measurement against a set of reference topics.

### API Layer

A FastAPI service (`api/main.py`) exposes the graph over HTTP:

| Endpoint | Method | Description |
|---|---|---|
| `/research` | POST | Start a run in the background; returns `202 {run_id, status:"queued"}` immediately |
| `/research/{run_id}` | GET | Status, the plan (dimensions / queries / complexity / rationale) for the approval screen, final report and evidence map |
| `/research/{run_id}/stream` | GET | Server-Sent Events: `node_started` / `node_finished` (with result sizes + audit entries), `awaiting_approval`, `done`, `failed`. Resume with `Last-Event-ID` |
| `/research/{run_id}/approve-plan` | POST | Resume from the human-approval interrupt (202, runs in the background). `409` unless the run is awaiting approval and not already executing |
| `/health`, `/ready` | GET | Liveness / readiness (ready checks the checkpointer) |

**Security:** every route except `/health` and `/ready` requires `X-API-Key` (or `Authorization: Bearer`) matching `API_KEY`; if `API_KEY` is unset the API is open and logs a warning (docker-compose makes it mandatory). Run-starting routes are rate-limited per client (`RATE_LIMIT_PER_MIN`), concurrent runs are capped (`MAX_CONCURRENT_RUNS`, over capacity returns `429`), inputs are length-limited, and `budget_mode` is one of `fast|balanced|thorough`. Optional `CORS_ORIGINS` / `ALLOWED_HOSTS` settings are documented in `.env.example`. Browsers' `EventSource` cannot send headers, so a frontend should read the stream with `fetch`.

The graph is built once at startup (`lifespan`). Runs execute on a bounded worker pool (`api/runs.py`); the event log is in-process memory (checkpoints in Postgres remain the source of truth, so after a restart old runs stay readable but lose their event history).

**Extractor safety:** URLs are fetched through `tools/safe_fetch.py`: http(s) only, public IPs only (loopback, private, link-local/cloud-metadata blocked), connection pinned to the validated IP, every redirect hop re-validated (max 3), proxies ignored, size and total-time caps.

---

## Project Structure

```
research_langgraph_setup/
|
|-- research_langgraph/
|   |-- __init__.py
|   |-- state.py                    # ResearchState TypedDict with reducer annotations
|   |-- main.py                     # Graph assembly, supervisor logic, run_research()
|   |
|   |-- agents/
|   |   |-- planner_agent.py
|   |   |-- search_agent.py
|   |   |-- search_merge.py
|   |   |-- extractor_agent.py
|   |   |-- extraction_merge.py
|   |   |-- validator_agent.py
|   |   |-- evidence_verifier_agent.py
|   |   |-- critic_agent.py
|   |   |-- synthesizer_agent.py
|   |
|   |-- routing/
|   |   |-- model_tiers.py          # Concrete model name registry (only place)
|   |   |-- model_router.py         # call_structured(), tier resolution, fallback chain
|   |
|   |-- tools/
|   |   |-- mcp_orchestrator.py     # Thin dispatch layer (HTTP or in-process)
|   |   |-- mcp_search_adapter.py   # Exa -> Tavily -> mock
|   |   |-- mcp_search_server.py    # Standalone FastAPI MCP search server
|   |   |-- mcp_extract_adapter.py  # PyMuPDF -> BeautifulSoup -> raw -> mock
|   |   |-- mcp_extract_server.py   # Standalone FastAPI MCP extract server
|   |   |-- mcp_registry.json       # Capability registry
|   |
|   |-- memory/
|   |   |-- checkpointer.py         # Postgres / in-memory checkpointer factory
|   |   |-- episodic_store.py       # Qdrant-backed episodic memory
|   |
|   |-- schemas/
|   |   |-- validated_outputs.py    # Pydantic schemas for all structured LLM outputs
|   |
|   |-- observability/
|   |   |-- langsmith_setup.py      # LangSmith tracer configuration
|   |   |-- execution_quality.py    # FULL / PARTIAL / DEGRADED tracking
|   |   |-- judge_eval.py           # LLM-as-judge offline evaluator
|   |
|   |-- api/
|       |-- main.py                 # FastAPI application
|
|-- docker/
|   |-- Dockerfile.mcp              # Container image for MCP tool servers
|
|-- Dockerfile                      # Main application image
|-- docker-compose.yml              # Wires app + MCP servers + Postgres + Qdrant
|-- requirements.txt
|-- .env.example
```

---

## Setup

**Prerequisites:**
- Python 3.11 or later
- PostgreSQL (optional; an in-memory checkpointer is used if `DATABASE_URL` is not set)
- Docker and Docker Compose (for containerized deployment)

**Install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Copy the environment file and fill in your keys:**

```bash
cp .env.example .env
```

---

## Configuration

All configuration is read from environment variables (or `.env`):

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Recommended | Groq key (primary LLM provider). Accepts several comma-separated keys: on 429 (rate-limited) or 401/403 (revoked) the next key is used automatically, and the bad key is skipped for a cooldown. Other errors do not rotate keys. |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Recommended | Gemini key (fallback provider); also accepts comma-separated keys, same failover behavior |
| `EXA_API_KEY` | Recommended | API key for Exa semantic search |
| `TAVILY_API_KEY` | Optional | API key for Tavily search (fallback to Exa) |
| `DATABASE_URL` | Optional | Postgres connection string for persistent checkpointing |
| `OFFLINE_MODE` | Optional | `1` forces offline placeholder extraction; unset = on only when no Exa/Tavily key is set |
| `LANGSMITH_API_KEY` | Optional | LangSmith API key; tracing is enabled automatically when set |
| `MCP_SEARCH_URL` | Optional | URL of a Dockerized MCP search server |
| `MCP_EXTRACT_URL` | Optional | URL of a Dockerized MCP extract server |
| `GROQ_MAX_CONCURRENCY` | Optional | Max concurrent Groq calls during fan-out (default: 4) |
| `GEMINI_MAX_CONCURRENCY` | Optional | Max concurrent Gemini calls during fan-out (default: 4) |

If no API keys are configured, the system falls through to deterministic mock functions at every layer and completes a full graph run offline. This is intentional: the graph is always runnable and testable without credentials.

---

## Running the System

**Command line (direct graph invocation):**

```bash
python -m research_langgraph.main "transformer architecture"
```

The run blocks at the human-approval interrupt if `auto_approve=False`. When called from the command line, `auto_approve=True` by default, so it runs end to end.

**API server:**

```bash
uvicorn research_langgraph.api.main:app --reload --port 8000
```

Start a research run (add `-H "X-API-Key: $API_KEY"` to every call when `API_KEY` is set):

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "transformer architecture", "budget_mode": "balanced"}'
```

Approve the research plan (when the run pauses at the approval gate):

```bash
curl -X POST http://localhost:8000/research/{run_id}/approve-plan \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

Poll for the completed report:

```bash
curl http://localhost:8000/research/{run_id}
```

---

## Docker Deployment

The `docker-compose.yml` starts five services:

| Service | Description |
|---|---|
| `api` | Main FastAPI application with the LangGraph pipeline (the only published port, on `127.0.0.1` by default) |
| `mcp-search` | Standalone search tool service (Exa / Tavily); internal network only |
| `mcp-extract` | Standalone extraction tool service (PyMuPDF / BeautifulSoup); internal only, receives no API keys |
| `postgres` | Postgres for LangGraph checkpointing; internal only |
| `qdrant` | Vector store for episodic memory (API-key protected); internal only |

Before the first run set `API_KEY`, `TOOL_SERVICE_TOKEN`, `QDRANT_API_KEY` and `POSTGRES_PASSWORD` in `.env` (compose refuses to start without them; see `.env.example`). Images are built from `requirements.lock` (hash-pinned, regenerate with `uv pip compile requirements.txt -o requirements.lock --python-version 3.12 --python-platform linux --generate-hashes`), run as a non-root user, and the base image is pinned by digest.

```bash
docker compose up --build
```

`MCP_SEARCH_URL`, `MCP_EXTRACT_URL` and `TOOL_SERVICE_TOKEN` are wired between containers automatically. The "MCP" services are plain HTTP tool services (the graph calls them directly from deterministic nodes, so the MCP protocol would add a server/handshake per call without a second consumer to justify it); if the tool services are down, the API falls back to the in-process adapters.

---

## Budget Modes

Pass `budget_mode` when starting a run to control the cost and depth tradeoff:

| Mode | Behavior |
|---|---|
| `fast` | Downgrades Synthesizer and Evidence Verifier to a smaller model; minimizes latency and token cost |
| `balanced` | Default; uses the standard tier assignment for every node |
| `thorough` | Upgrades Planner and Validator to a larger model; produces more thorough dimension decomposition and stricter source filtering |

Budget mode also influences the Supervisor's round and re-run budget, which is derived from the Planner's complexity estimate:

| Complexity | Search Rounds | Re-run Budget |
|---|---|---|
| simple | 1 | 0 |
| medium | 2 | 1 |
| complex | 2 | 3 |

---

## Execution Quality

Every completed run reports an `execution_quality` status in the evidence map:

| Status | Meaning |
|---|---|
| `FULL` | All model and tool calls used live providers; report and metrics are fully reliable |
| `PARTIAL` | One or more non-critical nodes fell back to a mock; report is still valid but coverage may be narrower |
| `DEGRADED` | One or more calls fell back to offline mocks; report content and quality metrics cannot be trusted |

A `DEGRADED` report is annotated with a prominent warning in the final output. The Critic can also issue a `retry_degraded` directive, which skips a new search round and sends the run directly back to the Synthesizer for a fresh pass, on the assumption that the underlying extracted documents are still valid and only the LLM judgment layer was compromised.

---

## Self-Healing Behavior

The system applies the same fallback philosophy at every layer:

- **Model calls:** Groq primary -> Gemini fallback -> deterministic mock
- **Search tools:** Exa -> Tavily -> offline mock results
- **Extraction tools:** PyMuPDF -> BeautifulSoup -> raw text fetch (HTML only) -> offline mock document (offline mode only; in live mode a failed fetch drops the source)
- **MCP servers:** HTTP dispatch to Dockerized server -> in-process adapter fallback
- **Checkpointer:** Postgres -> in-memory

No failure at any single layer halts the graph. Every fallover event is recorded in the audit log and surfaces in `execution_quality`, so degraded output is always distinguishable from full-quality output.
