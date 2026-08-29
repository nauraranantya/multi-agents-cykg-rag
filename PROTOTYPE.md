# AgCyRAG Prototype: Security Alert & Decision-Support System

Status doc for the prototype built on top of the original AgCyRAG repo (see
`README.md` for the original system). Companion to
`../defengraph_vs_agcyrag_analysis.md`, which this prototype directly acts
on (section references below point back to it). Plan file this was built
from: `misty-soaring-goose` (4-phase roadmap + Phase 0 prerequisite fix).

Last updated: 2026-08-26 (Sigma-rule MITRE enrichment + attack-graph
reconstruction added to §2.3a; security self-assessment against
AgenticCyOps added -- see `SECURITY_ASSESSMENT.md`).

---

## 1. What this prototype adds

The original AgCyRAG is a **manually-queried** multi-agent RAG system: an
analyst types a question, `src/run.py` invokes the LangGraph pipeline once,
and it queries a Neo4j graph that's assumed to already exist (built by an
external, no-code hosted tool -- nothing in the original repo builds it).

This prototype adds everything needed to turn that into an **automatic,
evaluated decision-support system**, and it is now fully running end-to-end
against real infrastructure (Neo4j AuraDB, OpenAI, the live SEPSES CSKG):

| # | Addition | Fixes / addresses |
|---|---|---|
| 0 | Lazy Neo4j/LLM init | `settings.py` crashed at import with no credentials at all (analysis §3.2.6) |
| 1 | Graph-construction + live stream | No graph-construction code existed anywhere (analysis §3.2.5); trigger was manual-only (comparison table). Originally paired with an auto-trigger gate -- superseded by #10 below, which decouples ingestion from investigation entirely |
| 2 | Structured synthesizer output + grounding check | Synthesizer was free-text with no way to verify citations (analysis §3.1.4, §5.4) |
| 3 | Temporal decay weighting on retrieval | AgCyRAG retrieval had no notion of recency at all |
| 4 | Ground-truth-based quantitative eval (no reference answers) | AgCyRAG had zero quantitative evaluation (analysis §3.5.15); scores against the dataset's own structured labels instead of narrative similarity, so no human-authored reference answer is needed |
| 5 | MCP RDF / CSKG live, bounded, and timeout-safe | `mcp-cskg-rdf` had no working config, an SSL trust gap, no result-size bounds, no ID-based technique lookup, and no timeout on its SPARQL calls (caused multi-minute hangs) |
| 6 | Wall-clock timeout on retrieval retry loops | `cypher_agent`/`vector_agent` retry only bounded *how many* attempts happen, not how long any single attempt could take -- same failure class as #5, just not yet observed there |
| 7 | LLM-as-a-Judge scoring (optional) | Structural ground-truth metrics (technique/escalation match) don't capture answer quality, faithfulness, or clarity -- added the validated rubric from Hamzić et al. 2604.11419v1 as a complementary scoring pass |
| 8 | MITRE-tag signal | Raw `rule_level` alone is a near-useless trigger predictor on real data (§3.2) -- `rule_level >= threshold OR native MITRE tag`, verified to raise recall 4.6x at the same default threshold. Originally an auto-escalation gate; now reused by #10 as an urgency-ranking signal instead |
| 9 | Non-blocking live ingestion | `trigger.py`'s live-stream loop writes each alert via `asyncio.to_thread` (the Neo4j driver call is synchronous) so a single ingest can't stall the event loop, even briefly |
| 10 | Ingestion/investigation split | Ingestion (`src/ingestion/`) used to auto-invoke the pipeline inline per alert, with no human in the loop. Now two separate processes: ingestion only ever stores alerts into the graph; a human-invoked `src/investigation/` service clusters stored alerts into ranked "activity cases" (time-session + cross-host shared-indicator merging), and a checkpointed LangGraph thread turns a selected case into an initial investigation followed by a multi-turn chatroom with real conversational memory |

---

## 2. Architecture

```mermaid
flowchart TD
    subgraph P1["src/ingestion/ (process 1 -- ingest only)"]
        AA["ait_ads.py<br/>AIT-ADS scenario sample<br/>(Zenodo 8263181, downloaded separately)"] --> LS["live_stream.py<br/>async replay, paced"]
        LS --> TR["trigger.ingest_stream()<br/>never invokes the pipeline"]
    end

    TR -->|every alert, unconditional| GL["graph_loader.py<br/>raw alert -> Neo4j"]
    GL --> NEO[("Neo4j AuraDB<br/>Host/IP/User/Alert/MitreTechnique<br/>+ Chunk/Document + vector/fulltext indices")]

    subgraph P2["src/investigation/ (process 2 -- human-invoked)"]
        CL["clustering.py<br/>fetch + time-session cluster +<br/>cross-host merge + urgency rank"]
        NAR["narrative.py<br/>case_context fact sheet"]
        SESS["session.py<br/>checkpointed_app (MemorySaver,<br/>thread_id = case_id)"]
        CL --> NAR --> SESS
        API["api.py (FastAPI)<br/>GET /cases, POST .../investigate,<br/>POST/GET .../chat"] --> CL
        API --> SESS
    end
    NEO -.queried by.-> CL
    SESS -->|turn 1: initial investigation,<br/>turn 2+: chatroom follow-up| APP[["checkpointed_app.ainvoke(...)"]]

    subgraph WF["AgCyRAG multi-agent pipeline -- src/graph/workflow.py"]
        GR["guardrails_agent"] --> VEC["vector_agent"]
        GR --> MCPA["mcp_rdf_agent"]
        VEC -.wall-clock timeout.-> VEC
        VEC --> RV["review_vector_answer"] --> CYP["cypher_agent"]
        CYP -.wall-clock timeout.-> CYP
        CYP --> RC["review_cypher_answer"] --> LA["log_analysis_agent"]
        LA --> MCPA
        MCPA --> SYN["synthesizer_agent<br/>SynthesizedReport + recommended_priority"]
        SYN --> GC["grounding_check<br/>non-LLM verify"]
        GC -->|ungrounded, retry once| SYN
        GC -->|grounded, or retried once| DONE(["answer + report"])
    end
    APP --> GR

    MCPA -.SPARQL, 41 tools, socket timeout.-> CSKG[("SEPSES CSKG<br/>w3id.org/sepses/sparql<br/>MITRE ATT&CK + CVE/CWE/CAPEC")]

    NEO -.queried by.-> CYP
    NEO -.queried by.-> VEC
    CYP -.temporal re-rank.-> TW["src/retrieval/temporal.py<br/>DefenGraph eq.5 decay"]
    VEC -.temporal re-rank.-> TW

    SESS --> ILOG[["data/investigations.jsonl<br/>(persistent audit trail)"]]

    subgraph EV["eval/"]
        RE["run_eval.py<br/>reads data/alerts.jsonl + alerts_ground_truth.json"] --> MET["metrics.py<br/>MITRE agreement, escalation P/R/F1,<br/>calibration, optional LLM-as-a-Judge"]
        ATT["analyze_trigger_threshold.py<br/>rule_level vs. ground truth, offline"]
    end
    RE -.invokes uncheckpointed app per alert, single-shot.-> RUNAPP[["src.run / eval: app.ainvoke(...)"]]
    RUNAPP --> GR
```

### 2.1 Original AgCyRAG (unchanged logic, only touched for wiring)

| File | Role |
|---|---|
| `src/agents/guardrails_agent.py` | Relevance check + routes to `log_analysis` or `cyber_knowledge` |
| `src/agents/vector_agent.py` | Full-text entity search + vector similarity search (Neo4j) |
| `src/agents/cypher_agent.py` | LLM-generated Cypher against the Neo4j schema |
| `src/agents/reflection_agent.py` | Rephrases the question on insufficient vector/cypher results |
| `src/agents/review_agent.py` | Judges sufficient/insufficient |
| `src/agents/log_analysis_agent.py` | Summarizes log findings, decides if CSKG lookup is needed |
| `src/agents/mcp_rdf_agent.py` | Queries the MITRE ATT&CK/CVE RDF graph via MCP |
| `src/graph/workflow.py` | LangGraph wiring of all of the above |
| `src/run.py` | CLI: one manually-typed question -> one pipeline run |

### 2.2 `src/config/settings.py`

Neo4j graph, vector index, and schema fetch are lazy (`_LazyProxy`,
`get_schema_escaped()`) -- connect on first real use instead of at import
time. Reads `NEO4J_AURA_DATABASE` (previously silently ignored -- this was
the actual cause of an early `DatabaseNotFound` error, not a provisioning
delay). `llm` always constructs successfully (placeholder API key when
unset). `_LazyProxy.resolve()` returns the real underlying object for
call-sites needing actual `isinstance` conformance --
`GraphCypherQAChain.from_llm(graph=...)` is pydantic-validated against
`GraphStore` and silently rejected the proxy at first real use (caught via a
live run, not by earlier offline testing, since offline testing never
exercised a Cypher call end-to-end).

### 2.3 `src/ingestion/`

| File | Role |
|---|---|
| `schema.py` | `Alert` Pydantic model (Wazuh-style fields, real MITRE IDs) |
| `ait_ads.py` | Loads a sample from the [AIT Alert Data Set](https://zenodo.org/records/8263181) (real Wazuh/Suricata/AMiner alerts) -- writes `data/alerts.jsonl` + `data/alerts_ground_truth.json`. Ground truth is derived from labeled attack-phase time windows (`labels.csv`), not a per-alert field |
| `graph_loader.py` | `ingest_alert()` / `load_all()` -- writes the structured entity graph **and** the Chunk/Document layer `vector_agent.py` expects (fulltext `entities` index, `vector`+`keyword` hybrid index) |
| `live_stream.py` | `stream_alerts()` -- async generator replaying `alerts.jsonl` in timestamp order |
| `trigger.py` | `ingest_stream()` -- ingests every alert immediately (never blocks: uses `asyncio.to_thread` for the sync Neo4j write) and **only** ingests; no investigation is triggered here. `_trigger_reason()` (`rule_level >= TRIGGER_SEVERITY_THRESHOLD` OR native MITRE tag) is still computed per alert, purely for an informational print -- it's kept around as a standalone signal because `eval/` imports it directly for comparison, and `src/investigation/clustering.py` reuses it as one input to urgency ranking (§2.3a) |
| `run_live.py` | CLI entrypoint wiring the two above |

### 2.3a `src/investigation/` (process 2 -- human-invoked)

The human-invoked replacement for the auto-trigger gate `trigger.py` used
to have: ingestion (2.3) never invokes the pipeline anymore, so nothing
does *unless* an analyst asks it to, via this package's FastAPI service
(`uv run -m src.investigation.run_api`, a separate process from ingestion).

| File | Role |
|---|---|
| `clustering.py` | `list_cases()`/`case_from_range()` -- fetches alerts from Neo4j (no persisted Case nodes; recomputed on demand every call) and clusters them into `Case`s: per-host time-session grouping (`CASE_SESSION_GAP_MINUTES`), then a Union-Find merge across hosts that share a source IP/dest user within `CASE_MERGE_GAP_MINUTES`. Every fetched alert is enriched with `sigma_matcher.py` matches before clustering. `score_urgency()` ranks cases 0-100 by reusing `_trigger_reason()` (noteworthy-alert count, now also true on a sigma-only match), max severity, case size, and `temporal.py`'s recency decay -- the same already-validated signals, just re-purposed from an auto-escalation gate into a ranking function |
| `sigma_rules/*.yml` + `sigma_matcher.py` | A small (10-rule), hand-curated set of real Sigma-format detection rules -- see the module's docstring for why public SigmaHQ rules don't transfer here (they target raw per-platform log fields our alert-level schema doesn't have) -- matched against `rule_description`/`full_log`/`decoder_name`/`rule_groups`/`agent_name` at investigation time (not ingestion, so rule changes apply retroactively without a re-ingest). Produces `Alert.sigma_mitre_id`/`sigma_matched_rules`, a second MITRE signal alongside the sensor's native `rule_mitre_id` (native tags cover only ~36% of AIT-ADS alerts). Supports a practical subset of Sigma's condition language (`and`/`or`/`and not`/`N of X*`), same scoping-down precedent KRYSTAL sets for its own Sigma-to-SPARQL translation |
| `attack_graph.py` | `build_attack_graph()` -- turns a case's member alerts into nodes (alert/host/IP/user/MITRE technique) and edges mirroring `graph_loader.py`'s real relation names, plus a derived `PRECEDES` edge chaining alerts that share an entity, in time order. The alert-granularity analogue of KRYSTAL's backward/forward chaining over its much lower-level per-syscall provenance graph. `root_cause_alert_id` = the alert with the most transitively-reachable downstream successors, ties broken by earliest timestamp -- exposed via `GET /cases/{case_id}/attack-graph` |
| `narrative.py` | `build_case_context()` -- renders a case + its member alerts into the static text fact sheet passed once into the pipeline as `state['case_context']`, now ordered by `attack_graph.py`'s reconstructed chain (falling back to severity ranking) with the likely root-cause alert called out explicitly, and native vs. sigma-matched MITRE tags labeled separately |
| `session.py` | Compiles `src.graph.workflow`'s uncompiled `workflow` StateGraph a *second* time with a `MemorySaver` checkpointer (`src/run.py`/`eval/` keep using the original uncheckpointed `app`, untouched). `start_investigation()` runs turn 1 (a fixed investigative directive + `case_context`) on a thread keyed by `case_id`; idempotent -- re-opening an already-started case returns the existing first turn instead of re-running the pipeline. `ask_followup()` runs later turns on the same thread with the analyst's literal message; `case_context` isn't resent since LangGraph's checkpointer carries the non-reducer field forward automatically. Every turn is also appended to `data/investigations.jsonl` (state itself is in-process-only and doesn't survive a restart) |
| `api.py` | FastAPI: `GET /cases`, `POST /cases/{case_id}/investigate`, `GET /cases/{case_id}/attack-graph`, `POST /investigations/time-range`, `POST/GET /investigations/{case_id}/chat`. Optional `INVESTIGATION_API_KEY`-gated (`X-API-Key` header) -- see `SECURITY_ASSESSMENT.md` |
| `run_api.py` | CLI entrypoint (uvicorn) |

Real conversational memory, not just message storage: `src/graph/state.py`'s
`case_context` field and `src/graph/workflow.py`'s `_render_conversation_context()`
thread the case fact sheet and prior turns into `guardrails_agent.py`,
`question_generation_agent.py`, and `synthesizer_agent.py`'s prompts (as
optional appended sections, same precedent as the existing `retry_note`
block) -- so a follow-up like "which host was that?" is actually answered
using what turn 1 already found, not re-derived as a context-free question.
`synthesizer_agent.py::SynthesizedReport` also gained a `mitigation_suggestions`
field, so an investigation's output is explicitly summary + diagnosis +
mitigation, not just one `final_answer` blob.

**Turn-efficiency routing**: guardrails skips its LLM call entirely on turn
1 (`state['skip_guardrails']` -- the input is a fixed directive over an
already-human-selected case, nothing to triage). On follow-up turns,
guardrails additionally decides `investigation_mode` -- `full_hypothesis`
(the default), `direct_question` (a narrow follow-up gets exactly one
targeted question instead of 3-5), or `answer_from_context` (the evidence
already gathered in a prior turn already answers this one, so
question_generation/dispatch_retrieval/review_evidence all skip
themselves and the synthesizer answers directly from
`question_results`/`log_cypher_context`/`log_vector_context`/
`mcp_rdf_context` -- plain, non-reducer state fields the checkpointer
already carries forward unchanged when nothing overwrites them, which is
what makes this a "cache" with no extra storage). Every node now also logs
its own output (not just that it ran) for observability, and
`TurnRecord.latency_seconds` records full per-turn wall-clock time.

**Guaranteed mitigation grounding**: `mitigation_suggestions` was previously
always free-generated by the synthesizer LLM from its own training
knowledge -- never actually pulled from `get_mitigations_for_technique`
(mcp-cskg-rdf/server.py), even when a real MITRE technique had already
been identified. `question_generation_node` now appends a deterministic
(non-LLM) cskg question asking for mitigations on every technique in
`state['known_mitre_techniques']` whenever there's at least one -- seeded
from the case's own native+sigma tags on turn 1
(`session.py::start_investigation`), grown by `synthesize_node` with
whatever `SynthesizedReport.mitre_techniques` each turn newly identifies,
and falling back to the single alert's own native tag for src/run.py/
eval/. Skipped (like all of question_generation_node) during
`answer_from_context` turns. Known gap this doesn't close:
`grounding_check.py` still doesn't verify `mitigation_suggestions` claims
against retrieved context the way it does `cited_entities`/
`mitre_techniques` -- a real CSKG-grounded mitigation lookup is now
guaranteed to happen, but the synthesizer isn't yet held to actually using
it over its own free generation.

**Evidence accumulates across turns, windowed**: `question_results` (and
`log_cypher_context`/`log_vector_context`/`mcp_rdf_context`/
`generated_question_for_rdf`, now DERIVED fresh from it each turn rather
than separately maintained) used to be wholesale REPLACED by every
`full_hypothesis`/`direct_question` turn -- none of `AgentState`'s fields
besides `messages` have a LangGraph reducer, so a node's return value
overwrites rather than merges. That meant an `answer_from_context` turn
could only ever see the most recent retrieval-performing turn's evidence,
not anything found earlier in the conversation -- a turn 2 on an unrelated
topic would silently erase turn 1's findings from what future turns could
reuse. `dispatch_retrieval_node` now tags each result with its turn number
and accumulates onto prior turns' `question_results` instead of replacing
it, windowed to the most recent `EVIDENCE_WINDOW_TURNS` (env-overridable,
default 5) turns so a long conversation's prompt sizes stay bounded.
`review_evidence_node` was updated in lockstep -- it now only reviews
entries missing a `review` (this turn's new questions), reusing prior
turns' verdicts as-is rather than wastefully (and non-deterministically)
re-reviewing them every turn. `_render_hypothesis_summary` tags each line
with its turn number so a multi-turn summary stays legible.

**Query-quality pointers**: a live run surfaced four distinct causes behind
consistently empty/failed retrieval results, diagnosed via `tools_used` in
the `[mcp_rdf_agent]` log line above and the raw Cypher error text -- fixed
at the prompt level rather than papered over with more retries:
- `question_generation_agent.py` now states explicitly what `cypher`
  actually has data for (Host/IP/User/Alert/MitreTechnique correlation --
  nothing about authorization, config/IDS-rule changes, or asset
  ownership) and what `cskg` does/doesn't cover (technique/CVE reference
  meaning, never IP/host-specific correlation) -- previously it generated
  plausible SOC-analyst questions the schema simply has no data to answer,
  which isn't a retrieval failure, it's an unanswerable question.
- `mcp_rdf_agent.py`'s system prompt now explains that most cskg tools
  (`get_mitigations_for_technique`, `get_techniques_by_tactic`, etc.)
  match on technique/entity NAME text, not MITRE ID -- passing a bare ID
  silently returns nothing every time, which is exactly what happened to
  the guaranteed mitigation question above (`get_mitigations_for_technique`
  called 3 times, once per ID, zero results each time). Both the general
  system prompt and that specific guaranteed question now spell out the
  correct sequence (resolve ID -> name via `get_technique_by_id` first, or
  use `text_to_sparql` to filter by ID directly).
- `cypher_agent.py` gained a rule for the specific Cypher syntax error
  observed: a `RETURN`/`WITH` using `DISTINCT` or an aggregate drops every
  variable not explicitly returned, so a later `ORDER BY` referencing an
  earlier MATCH variable is invalid -- a common LLM-generated-Cypher
  mistake, not caught by the existing 10 rules.
- `cypher_agent.py` also gained an upfront "what data does this question
  actually need" framing step, plus a rule specifically about
  `MitreTechnique.tactic`: it's only ever one of the ~14 standard MITRE
  tactic names (native-tagged alerts only, most don't have one) -- never a
  hypothesis's own wording, so `WHERE toLower(m.tactic) CONTAINS
  "scanning"` was guaranteed to return nothing regardless of whether
  matching data existed. A specific technique should filter by `m.id`
  (the structured identifier); an open-ended question should retrieve the
  alert's own `title`/`full_log` broadly instead of guessing a keyword
  filter -- added as a new example (#6) alongside the existing five,
  showing the wrong pattern next to the right one.

**Escalation triage removed**: `guardrails_agent.py` used to also decide
`should_escalate` ("should this auto-triggered alert be investigated at
all") -- vestigial once ingestion/investigation split into two processes:
nothing auto-triggers the pipeline anymore, an analyst always invokes
investigation deliberately, so there was nothing left to triage. Removed
from `GuardrailsOutput`, `AgentState`, and `decide_after_guardrails`, which
now routes purely on relevance. `alert_triage_agent.py` (a standalone
classifier this decision used to be compared against, never wired into the
live pipeline) and its sole consumer `eval/evaluate_alert_triage.py` were
removed in the same pass. `eval/run_eval.py`'s escalation-confusion-matrix
metric is unaffected in what it *reads* (still `synthesized_report.
recommended_priority`), but every alert it scores now always reaches the
synthesizer for a real judged priority, rather than some being
short-circuited to a hardcoded "ignore" by the removed gate -- a
behavioral change worth knowing about if comparing against pre-cleanup
eval numbers.

**Known limitation**: `case_id` is a hash of its exact member-alert set at
clustering time. A still-growing activity cluster (new alerts keep landing
on the same host/indicators after an analyst already opened it) hashes to
a *different* case_id on the next `GET /cases`, surfacing as a "new" case
rather than extending the same investigation thread. Accepted for v1 to
keep clustering stateless -- no persisted Case identity to keep in sync
with a live ingestion stream running as a separate process.

**Security**: `SECURITY_ASSESSMENT.md` applies AgenticCyOps' (Mitra et al.
2026) own coverage-matrix evaluation method to this codebase's own
multi-agent/MCP/persistent-memory architecture. Two concrete fixes came out
of that pass: `src/config/settings.py`'s `ReadOnlyNeo4jGraph` (a dedicated,
separate Neo4j connection that rejects any write/schema-mutating clause
before it reaches the database -- `cypher_agent.py`'s LLM-generated Cypher
previously had no technical restriction, only a prompt instruction not to
write/delete), and the `INVESTIGATION_API_KEY` gate on `api.py` above.

### 2.4 Structured output + grounding

- `src/agents/synthesizer_agent.py`: `SynthesizedReport` (8 report fields + `cited_entities`, `mitre_techniques`, `confidence`, `recommended_priority` [ignore/monitor/escalate]) via `llm.with_structured_output(...)`.
- `src/agents/grounding_check.py`: regex-based, non-LLM check that every cited entity/MITRE ID/IP appears in retrieved context.
- `src/graph/workflow.py`: `grounding_check` node after `synthesizer`, one bounded retry on failure.

### 2.5 `src/retrieval/temporal.py`

`temporal_weight(node_ts, query_ts, alpha_per_hour=0.05)` = DefenGraph eq. 5,
ported from `../defengraph_recreation/src/llm_stages.py`. Re-ranks
`cypher_agent.py` / `vector_agent.py` results by recency. Missing timestamp
= weight 1.0, matching DefenGraph's immutable-SKG-entity convention.

### 2.6 `mcp-cskg-rdf/` (MITRE ATT&CK + CVE server)

Live and working against the public SEPSES CSKG (`https://w3id.org/sepses/sparql`),
per the original AgCyRAG paper's own methodology (§4.1) -- no local `.ttl`
file needed. Fixes made over the course of this prototype:

1. **`browser_mcp.json`** created at the repo root, pointing at the actually-bundled
   server (README previously documented a different external repo with a wrong flag).
2. **`truststore` injection** in `server.py` -- the endpoint's TLS cert chain
   (a recent Let's Encrypt intermediate) isn't trusted by every installed
   `certifi` snapshot, even though it's valid (curl/macOS accept it fine).
   Confirmed not a systemic issue (a plain `google.com` request worked)
   before diagnosing and fixing.
3. **Result-size bounds** in `format_sparql_results()` -- previously
   unbounded; a single broad keyword call could return 50 rows with full
   multi-paragraph descriptions each, bloating context and latency. Now
   capped (400 chars/description, ~6000 chars total). Discovered live, not
   theoretically -- watched a real run stall on exactly this.
4. **`get_technique_by_id` tool added** -- none of the original 40 tools
   could look up a technique by its ATT&CK ID (e.g. "T1505.003"), only by
   name/label text. Fixed by following the existing `get_cve_by_id` pattern;
   verified the agent picks it correctly on the first try post-fix.
5. **`get_techniques_by_platform` SPARQL type error fixed** -- `?platform`
   is a URI resource, not a literal; `LCASE()` needs `STR()` wrapping first.
6. **`get_mitigations_for_technique` duplicate-row explosion fixed** -- the
   underlying CSKG has duplicate triples and parallel canonical/slug URIs
   for the same entity; `SELECT DISTINCT` over label columns only (dropping
   URI columns) collapsed ~76 near-identical rows for "Valid Accounts" down
   to the 12 genuine distinct mitigations.
7. **Socket-level timeout** (`socket.setdefaulttimeout(30)`) -- `rdflib`'s
   `SPARQLConnector.query()` calls `urlopen()` with no timeout on any code
   path (confirmed directly in its source), and passing `timeout=` to
   `SPARQLStore()`'s constructor doesn't help either since it's never
   extracted from `self.kwargs`. This caused two real hangs (37 min and 11
   min, confirmed via `ps aux` CPU-time deltas showing near-zero CPU over
   long wall-clock time) before being root-caused and fixed. Verified live:
   a normal query still succeeds (~3s); a deliberately-unreachable host
   correctly raises after the timeout instead of hanging.

### 2.7 Retrieval timeout (`src/graph/workflow.py`)

The MCP hang above (§2.6.7) is one instance of a broader gap: `max_iterations`
(default 3) bounds how many times `cypher_agent`/`vector_agent` retry, but
nothing bounded how long any *single* attempt could take. A slow or stuck
LLM call or Neo4j query could stall the whole pipeline indefinitely --
exactly the failure class Hamzić et al. (2604.11419v1) document as GraphRAG's
worst-case behavior (Case 3: a 2,348s/39-minute hang on an unanswerable
query, iteratively retrying invalid Cypher property names instead of ever
concluding "not found").

Fixed by running every `cypher_agent`/`vector_agent`/`mcp_rdf_agent` call in
a worker thread (`asyncio.wait_for` for the async MCP node) with a hard
wall-clock deadline (`NODE_CALL_TIMEOUT_SECONDS`, default 60s, env-overridable).
A timeout is treated as an ordinary retrieval failure -- the existing
except blocks already degrade gracefully to "no context found" and let
reflection/`max_iterations` take over, so this needed no new error-handling
path, just a bound on how long any one attempt can occupy before that
existing path kicks in. Verified with a synthetic fast/slow-function test:
a 2s function call under a 1s timeout raises cleanly instead of blocking.

### 2.8 LLM-as-a-Judge scoring (`eval/metrics.py::score_llm_judge`)

Structural ground-truth metrics (MITRE technique match, escalation match)
only check whether specific facts line up -- they say nothing about whether
the prose answer is well-formed, faithful to the retrieved evidence, or
hallucinates unrelated detail. Added the validated 4-criterion weighted
rubric from Hamzić et al. 2604.11419v1 Sec.3.11: Agreement (weight 4),
Adequacy (weight 3), Faithfulness (weight 2), Clarity (weight 1), each
scored 0-5 by an LLM judge, weighted total out of 50. Their own post-hoc
validation (N=2,640 judgments) found Agreement/Adequacy/Faithfulness form a
tightly coupled correctness cluster (r=0.91-0.98) while Clarity stays
substantially less correlated (r=0.43-0.50) -- confirming it captures a
genuinely orthogonal quality dimension (a fluent-but-wrong answer scores
high on Clarity, low on the rest) rather than just restating the others.

Since this dataset has no human-authored narrative reference answer,
`eval/run_eval.py::build_baseline_answer()` synthesizes the minimal factual
statement the structural ground truth actually supports (attack-phase
membership + MITRE technique, if any) for the judge to compare against --
not a full analyst writeup, just the facts a correct answer must not
contradict. One extra LLM call per scored alert, so it's opt-in
(`--llm-judge` flag / `run_eval.sh --llm-judge`) rather than always-on.

### 2.9 `eval/`

| File | Role |
|---|---|
| `run_eval.py` | Reads `data/alerts.jsonl` + `data/alerts_ground_truth.json` locally (no API calls to load). By default runs *every* sampled alert through `app.ainvoke` regardless of level, measuring the LLM's own judgment quality against ground truth (`mitre_technique`/`should_escalate`/`confidence`). `--llm-judge` also runs `score_llm_judge` per alert. `--gate-filter` instead only runs alerts passing `TRIGGER_SEVERITY_THRESHOLD` (matching real production behavior) and reports **end-to-end** precision/recall (gate + LLM combined, decomposed into gate recall × LLM-conditional recall) against all true attack-phase alerts in the sample, not just the ones that reached the LLM |
| `metrics.py` | Four metric classes, each mapping onto an established evaluation approach rather than something invented for this project: MITRE technique agreement (TRAM/TTPrint/TTPXHunter-style precision/recall/F1), escalation confusion matrix (SOC alert-triage-LLM-benchmark-style), calibration-by-confidence bucketing (OpenSec-style), and LLM-as-a-Judge (§2.8) |
| `analyze_trigger_threshold.py` | Offline (no LLM/DB calls): sweeps `TRIGGER_SEVERITY_THRESHOLD` against `rule_level` vs. ground-truth `should_escalate` in the current local sample -- see §3.3 finding below |

---

## 3. Findings

### 3.1 Confidence calibration fix

Confidence was found to be "high" on all 25 predictions in an early
evaluation run, including every wrong escalation -- confidently wrong, not
hedged, and uninformative for the calibration check. Root cause: the field
was purely LLM self-assessed with no structural constraint. Fix in
`src/graph/workflow.py`: `_context_richness()` deterministically counts how
many of the 3 possible evidence sources (Cypher/vector/CSKG) returned
substantive (non-placeholder, non-error) content, and `_cap_confidence()`
caps the LLM's stated confidence at what that count can support (0 sources
-> at most "low", 1 -> at most "medium", 2-3 -> "high" allowed). Only ever
lowers confidence, never raises it. Unit-verified directly (0/1/3 sources ->
low/medium/high cap) and confirmed capping live on a genuinely low-evidence
alert (capped to "medium" instead of defaulting to "high").

This is independently validated by Hamzić et al. 2604.11419v1's concept of
*structural hallucination* (Table 25): a fluent answer that's formally
consistent with retrieved data but unsupported because the schema/graph
didn't actually cover the question -- exactly the failure mode this fix
targets, arrived at independently before that paper was read.

### 3.2 Trigger threshold is not discriminative on real AIT-ADS severity data

**First pass** (87-alert sample): `eval/analyze_trigger_threshold.py` found
every single sampled alert was `rule_level=3`, with `TRIGGER_SEVERITY_THRESHOLD=7`
catching zero of 50 genuine attack-phase alerts. This turned out to be a
real bug, not a property of the dataset: `src/ingestion/ait_ads.py`'s
`to_alert()` derived host identity only from `predecoder.hostname` or
`data.dest_ip` -- but almost every high-severity alert in AIT-ADS (across
all 8 scenarios, confirmed by downloading and checking the full Zenodo
archive) is a Suricata/snort-decoded network alert that has neither field.
`agent.ip` (the actual internal host that generated the alert) is present
on 100% (778/778 checked) of those previously-unrepresentable level>=7
alerts and is now used as a fallback -- fixed in `to_alert()`.

**Second pass** (178-alert sample, post-fix): real severity variety is now
visible (levels 3/5/6/8/10/11 present). `analyze_trigger_threshold.py` was
extended to report F1 and F2 (recall-weighted 4x, a rough proxy for
security alerting's usual cost asymmetry -- a missed attack is normally far
more expensive than an analyst dismissing a false alarm) per threshold, and
to flag Pareto-dominated thresholds automatically. Findings on this sample:
`rule_level=6` is strictly dominated (worse precision *and* recall than
`rule_level=5` simultaneously -- an artifact of a batch of generic "IDS
event." alerts sitting at exactly that level) and should never be chosen.
Both F1 and F2 pick `rule_level>=3` (essentially no severity filtering) as
best among pure-severity cutoffs, since severity alone loses recall faster
than it gains precision at every other candidate threshold. `rule_level>=7`
(the production default) recall was 0.039 on this sample.

**Fix landed**: `src/ingestion/trigger.py`'s gate is now `rule_level >=
TRIGGER_SEVERITY_THRESHOLD OR alert.rule_mitre_id` (native MITRE ATT&CK tag
present) rather than severity alone -- verified to raise recall to 0.180
(4.6x) at the same threshold=7 default, with 38 of 43 triggered alerts
reaching the pipeline via the MITRE-tag path specifically, not severity.
Still open: `rule_level>=5 OR has_mitre_tag` scored meaningfully better yet
in testing (F1 0.747 vs the deployed combo's 0.269) -- whether to also
lower the default threshold remains an open, not-yet-made call (a policy
question about relative cost of missed attacks vs. false alarms, not a
statistical one). See §5.

---

## 4. What's fully working (live-verified, not just offline)

- Full pipeline runs end-to-end against real Neo4j AuraDB + OpenAI + the live SEPSES CSKG.
- AIT-ADS alert sample ingested and queryable, with genuine cross-alert host correlation (a handful of hosts hit repeatedly, unlike single-alert-per-host data sources tried earlier).
- Cross-alert correlation confirmed in real output (e.g. an alert about outbound traffic correctly surfaced an *earlier* alert's SSH login and discovery activity from the same incident -- not just the triggering alert's own text).
- Grounding check passes/fails correctly on live runs, not just unit tests.
- Temporal re-ranking, structured synthesizer output, and the full MCP/CSKG path all verified together in the same live run.
- Confidence calibration cap verified both in isolation and live.
- MCP SPARQL timeout and retrieval-loop timeout both verified live (normal calls unaffected; simulated hangs correctly time out instead of blocking).

## 5. What still needs to be done

1. **Run a full clean eval** now that the retrieval-timeout fix and LLM-as-a-Judge scoring are both in place -- the last two attempts were interrupted mid-run (once by a since-fixed MCP hang, once manually) before either fix landed.
2. **Decide whether/how to act on the escalation precision finding** -- tune the triage prompt, or treat high-recall/low-precision as the intentionally-correct security posture (never miss a real threat) and report it as-is. (An early over-escalation pattern was observed live: alerts get flagged as escalate-worthy based on surface language -- "Authentication", mail-server context -- rather than actual severity signal, even on plain, level-3, non-attack-phase events like routine Dovecot logins.)
3. ~~**Trigger threshold needs a real fix**~~ **Partially done**: `trigger.py`'s gate is now `rule_level >= TRIGGER_SEVERITY_THRESHOLD OR alert.rule_mitre_id` (native MITRE ATT&CK tag present), not severity alone. Verified on the current sample: recall rose from 0.039 (severity-only) to 0.180 (4.6x) with `TRIGGER_SEVERITY_THRESHOLD` left at its default of 7 -- 38 of 43 triggered alerts now get through via the MITRE-tag path, not severity. Still open: `eval/analyze_trigger_threshold.py`'s Pareto/F-beta analysis found `rule_level>=5 OR has_mitre_tag` scores meaningfully better still (F1 0.747 vs 0.269 for the tag-plus-level-7 combo actually deployed) -- whether to also lower the default threshold is a separate, not-yet-made decision (see §3.2's F1 vs F2 discussion: it depends on how costly a missed attack is judged to be relative to analyst review time, a policy call, not a statistical one). A lightweight frequency/correlation pre-filter beyond these two signals remains unexplored.
4. **SPARQL-side `LIMIT` clauses** on the handful of MCP tools still missing one (defense-in-depth beyond the output-bounding fix in §2.6.3, lower urgency now that the actual observed problem is fixed).
5. Analysis doc's other proposals (§5.1 three-layer KG, §5.6 adversarial-robustness eval, §5.7 SOAR baseline) remain out of scope for this prototype.

## 6. Known limitations / deviations (for the thesis writeup)

- **Single-scenario sample.** Only `russellmitchell` (one of AIT-ADS's 8 scenarios) has been downloaded and evaluated so far; findings like §3.2's flat-severity pattern are confirmed for this scenario specifically, not yet cross-checked against the other 7.
- **`alpha_per_hour=0.05`** in temporal decay is a free parameter, same as DefenGraph's own paper -- not tuned against ground truth.
- **Grounding check is regex/substring-based, not semantic.** Only catches claims using recognizable entity syntax; a hallucination phrased without those patterns won't be caught (analysis §5.4's stated non-LLM tradeoff).
- **MITRE agreement metric is recall-only against a single labeled technique per alert.** A bot correctly citing additional, genuinely-relevant techniques the dataset didn't label would be penalized on precision if that were computed -- a known limitation of single-label ground truth, not of the bot. It's also only computed over the ~36% of alerts that carry a native MITRE tag at all.
- **Escalation ground truth is derived purely from attack-phase timestamp windows.** It may depend on signals the bot isn't given (e.g. base-rate context, whether an IP/host is already known-bad) -- observed false-positive patterns may partly reflect this rather than pure reasoning failure; not disentangled in this prototype.
- **Retrieval timeout (§2.7) is a coarse wall-clock bound (default 60s), not a smarter retry/backoff strategy.** A legitimately slow-but-eventually-successful call past the deadline is treated the same as a genuinely stuck one.
