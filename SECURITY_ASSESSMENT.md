# Security Self-Assessment: AgCyRAG as a Multi-Agent System

Applies AgenticCyOps' (Mitra et al. 2026, arXiv:2603.09134) own evaluation
method -- decompose attack surfaces by component/coordination/protocol
layer, map them against a small set of defensive principles, then count
what's actually mitigated vs. open -- to this codebase's own architecture.
AgenticCyOps' threat model (LLM multi-agent systems built on tool
orchestration + persistent memory, often over MCP) is not a hypothetical
comparison here: it's a direct description of what this repo already is
(`src/agents/mcp_rdf_agent.py` is an MCP client; `src/agents/cypher_agent.py`
executes LLM-generated queries against a live graph; `src/investigation/
session.py`'s checkpointed pipeline is exactly the "persistent multi-turn
memory in a MAS" surface AgenticCyOps calls out as under-addressed
elsewhere in the literature).

This is a design-time self-assessment (structural analysis, like
AgenticCyOps' own evaluation), not a penetration test or adversarial
simulation -- the same limitation AgenticCyOps' own Discussion section
names for its evaluation.

## 1. Surfaces in this codebase

| Surface | What it is | AgenticCyOps category |
|---|---|---|
| `guardrails_agent.py` / `question_generation_agent.py` / `synthesizer_agent.py` | LLM calls reasoning over alert/case text, including analyst-controlled follow-up messages | Component-level (Perception/Cognitive) |
| `cypher_agent.py` | LLM *generates and executes* Cypher against the live Neo4j graph (`GraphCypherQAChain`, `allow_dangerous_requests=True`) | Tool orchestration (highest-risk surface in this repo) |
| `vector_agent.py` | Hybrid vector+keyword search, read-only by construction (LangChain's `Neo4jVector` similarity search issues no write Cypher) | Tool orchestration (low risk) |
| `mcp_rdf_agent.py` | MCP client, `MCPAgent` calling `mcp-cskg-rdf`'s ~41 tools against the **public** SEPSES SPARQL endpoint | Protocol-level (MCP) / Tool orchestration |
| `src/investigation/session.py` | `checkpointed_app` -- a `MemorySaver`-backed LangGraph thread per `case_id`, holding multi-turn conversation state (`case_context`, `messages`, prior reports) | Memory management |
| `src/investigation/api.py` | FastAPI service; every route can start an investigation (LLM calls) or read/write into any case's chat thread | Protocol-level (the interface itself) / Memory access |
| `grounding_check.py` | Non-LLM, deterministic: verifies the synthesizer's cited entities actually appear in retrieved context | Existing defense (see below) |

## 2. Coverage matrix

Columns are AgenticCyOps' five principles: **P1** Authorized Interface,
**P2** Capability Scoping, **P3** Verified Execution, **P4** Memory
Integrity & Synchronization, **P5** Access Control with Data Isolation.
`✓` = directly mitigated in this codebase today, `◦` = partially/secondary,
blank = open gap.

| Surface | Risk | P1 | P2 | P3 | P4 | P5 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| `cypher_agent.py` generated-Cypher execution | LLM-generated Cypher executes with no built-in write/delete restriction; a prompt-injected `full_log`/`rule_description` could attempt `DETACH DELETE`/`SET`/etc. | | **✓** `ReadOnlyNeo4jGraph` (`src/config/settings.py`) rejects any write/schema clause before it reaches Neo4j -- a real technical control, not the prompt's "don't write" instruction alone | ◦ rejection happens pre-execution, not a full authorize-then-verify pipeline | | |
| `mcp_rdf_agent.py` MCP tool calls | Public, read-only SEPSES endpoint -- worst case is wasted queries/latency, not data loss on *our* infrastructure | | ◦ inherently scoped by the endpoint being read-only and public | | | |
| Investigation API (`api.py`) | No caller identity at all before this pass -- anyone reaching the service could read/write into any `case_id`'s chat | **✓** `INVESTIGATION_API_KEY` + `X-API-Key` dependency on the whole app | | | | ◦ a single shared secret, not per-analyst identity -- see gaps below |
| `session.py`'s `MemorySaver` state | Single in-process store; `case_id` is the only namespacing between investigation threads | | | | | ◦ thread_id-keyed, but any authenticated caller can address any thread_id -- no ownership check |
| `synthesizer_agent.py` output | An LLM could cite entities/techniques not actually present in retrieved evidence (hallucination) | | | **✓** `grounding_check.py` -- non-LLM, deterministic verification with one bounded retry | | |
| `guardrails_agent.py` escalation decision on a follow-up | A malicious analyst message (or, more realistically, a benign one an LLM can't parse) could mis-triage a continued investigation | | | ◦ prompt instructs "a follow-up should almost always keep escalating," not enforced structurally | | |

*Coverage note*: every attack vector identified has at least one mitigation
or documented gap above; no vector is silently unaddressed. Two of five
principles (P2, P3) have concrete technical controls in this codebase
today; P1 has one (the API key); P4 and P5 are the largest open area.

## 3. What's genuinely mitigated

- **Capability scoping on `cypher_agent.py`** (this pass): a dedicated
  `ReadOnlyNeo4jGraph` connection, separate from the shared write-capable
  `graph` singleton ingestion uses, rejects any Cypher containing
  `CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD CSV|CALL apoc.<write
  family>` before it reaches Neo4j. Tested directly (no live DB needed --
  the check runs before the network call): 3 representative safe queries
  pass through unchanged, 7 representative write/schema queries are
  rejected.
- **Authorized interface on the investigation API** (this pass): optional
  `INVESTIGATION_API_KEY` gate via FastAPI `Depends`, applied globally.
  Verified live: no key → 401, wrong key → 401, correct key → 200.
- **Verified execution on synthesizer output** (pre-existing):
  `grounding_check.py` already does, for LLM *output*, what AgenticCyOps'
  P3 asks for on tool *execution* -- reject/retry once on ungrounded
  citations rather than trusting the model's own confidence.

## 4. Open gaps (not addressed this pass, documented rather than silently left)

- **No per-analyst identity.** `INVESTIGATION_API_KEY` is a single shared
  secret -- it answers "is this caller authorized to use the service at
  all," not "which analyst is this, and do they own this case." Real
  per-user RBAC would need an actual multi-user model, which doesn't exist
  in this prototype (single-analyst-at-a-time is the implicit assumption
  throughout `session.py`).
- **No consensus validation on any tool call.** Every LLM call in this
  pipeline (guardrails, question generation, cypher generation, synthesis)
  is trusted on a single pass; AgenticCyOps' P3 in its full form assumes a
  second, independent validator agent. Out of scope for a single-analyst
  prototype pipeline -- would meaningfully change the architecture (and
  cost/latency) for a benefit that matters more at genuine multi-tenant
  scale.
- **`MemorySaver` is unbounded and process-local.** Any authenticated
  caller can address any `case_id` thread (no ownership check beyond
  knowing/guessing the id, which itself is a content-derived hash, not a
  secret). State is lost on process restart by design (see `session.py`'s
  own docstring) -- acceptable for a prototype, not for a durable service.
- **`guardrails_agent.py`'s "always escalate on a follow-up" rule is a
  prompt instruction**, not a structural guarantee, mirroring the same
  class of gap the `cypher_agent.py` fix above just closed for Cypher
  execution specifically. Lower priority: worst case here is a missed
  investigation, not data loss.

## 5. Relationship to KRYSTAL

Unlike AgenticCyOps, KRYSTAL (Kurniawan et al. 2022) is not itself a
security threat model for the *system running it* -- it's the detection
technique (see `PROTOTYPE.md` §2.3a) this codebase's sigma-rule enrichment
and attack-graph reconstruction are modeled on. No overlap in scope with
this document; noted here only so the two papers' roles in this repo
aren't conflated.
