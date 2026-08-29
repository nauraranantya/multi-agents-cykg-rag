
# Human-Gated Multi-Agent Security Investigation

A knowledge-graph-grounded RAG system for security alert investigation, built around two separate
processes: **ingestion** (continuous, deterministic, stores alerts and never triggers anything on its
own) and **investigation** (always human-invoked -- an analyst reviews ranked activity cases, launches
a hypothesis-driven multi-agent investigation on one, and continues in a checkpointed multi-turn
chatroom with real conversational memory, not a stateless Q&A loop). Every retrieval-backed claim is
grounded and verified against actual retrieved evidence, and stated confidence is capped by how much
evidence genuinely supports it, rather than the model's own unconstrained self-assessment.

**Built on and credited to [AgCyRAG](https://github.com/sepses/multi-agents-cykg-rag)** (Kurniawan et
al., *"AgCyRAG: an Agentic Knowledge Graph based RAG Framework for Automated Security Analysis,"*
RAGE-KG 2025 workshop @ ISWC) -- this repo started as a fork of `sepses/multi-agents-cykg-rag`
(`git remote -v` still shows it as `upstream`) and originally reused its manual, single-question,
LangGraph-orchestrated pipeline (`src/run.py` still runs that original mode directly). Everything
described below beyond that -- graph construction (none existed in the original repo), hypothesis-driven
question generation, structured/grounded output, confidence calibration, and the entire ingestion/
investigation split with case clustering and multi-turn memory -- was added on top of that foundation.
See `PROTOTYPE.md` for the full architecture, design rationale, and known limitations, and
`CONTRIBUTIONS.md` for a precise accounting of what's original here versus inherited from AgCyRAG and
related prior work.

## Core Components

- **Multi-Agent System (LangGraph)**: orchestrates the investigation pipeline -- a relevance gate, a
  hypothesis-driven question generator (proposes several competing explanations for an alert/case, not
  one fixed interpretation), concurrent retrieval dispatch across three sources, an evidence reviewer,
  a structured-output synthesizer, and a non-LLM grounding check with a bounded retry.
- **Neo4j Knowledge Graph**: stores the ingested alert data (hosts, IPs, users, alerts, MITRE
  technique associations) as a structured entity graph, plus a Chunk/Document layer for hybrid
  vector+keyword search over alert narratives -- built by `src/ingestion/graph_loader.py`.
- **MCP RDF Explorer**: an MCP server (`mcp-cskg-rdf/`) providing tool access to the public SEPSES
  CSKG (MITRE ATT&CK, CVE/CWE/CAPEC), including a real LLM-driven text-to-SPARQL fallback tool for
  questions the ~40 fixed tools don't directly cover.
- **`src/investigation/`**: clusters ingested alerts into ranked "activity cases" (time-session +
  cross-host shared-indicator merging), runs an initial hypothesis-driven investigation on a selected
  case, and opens a checkpointed multi-turn chatroom for analyst follow-up -- with real conversational
  memory (not just message storage) and turn-windowed evidence accumulation. Exposed as a FastAPI
  service, a separate process from ingestion.

## Setup and Installation

Make sure that you already have uv installed on your desktop, if not then here's the installation guide : https://docs.astral.sh/uv/getting-started/installation/

```bash
  git clone <this-project>
  cd <this-project>
  uv sync
```

The MCP RDF server bundled in this repo (`mcp-cskg-rdf/`) is the one that actually
gets used -- earlier versions of this README pointed at a different, external
repo with a `--triple-file` flag, which is wrong.

- Install its deps into the same venv the rest of the project uses:
  `.venv/bin/pip install -r mcp-cskg-rdf/requirements.txt` (or
  `uv pip install -r mcp-cskg-rdf/requirements.txt --python .venv/bin/python`)
- Per the AgCyRAG paper (Sec.4.1), the CSKG this queries is the **SEPSES CSKG**,
  publicly accessible via its own SPARQL endpoint -- no local `.ttl` file needed.
  Create `browser_mcp.json` at the root of `multi-agents-cykg-rag/` (absolute
  paths -- `mcp_rdf_agent.py` launches this as a subprocess):
```json
{
  "mcpServers": {
    "rdf_explorer": {
      "command": "/absolute/path/to/multi-agents-cykg-rag/.venv/bin/python",
      "args": ["/absolute/path/to/multi-agents-cykg-rag/mcp-cskg-rdf/src/mcp-cskg-rdf/server.py",
                "--sparql-endpoint", "https://w3id.org/sepses/sparql"]
    }
  }
}
```
  (A local `.ttl` file also works via `--rdf-file <path>` instead, if you'd
  rather not depend on the live public endpoint.)
- Note: the public SEPSES endpoint's TLS certificate chain isn't trusted by
  every installed `certifi` snapshot, even though it's valid (curl/macOS both
  accept it fine) -- `server.py` injects `truststore` at startup (verifies
  against the OS trust store instead) specifically to handle this.

Make sure again that the .env file is filled !!!

`src/config/settings.py` only ever reads the `NEO4J_AURA*` variables below (a
managed Neo4j AuraDB instance) -- there is no local/self-hosted Neo4j path in
the code, regardless of what earlier versions of this README implied.
`LANGCHAIN_*` are optional (LangSmith tracing); leave them blank to skip
tracing. `INVESTIGATION_API_KEY` is optional too -- unset means the
investigation API (below) is open, matching local-dev convention; set it to
require an `X-API-Key` header on every request.

```bash
OPENAI_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_ENDPOINT=
LANGCHAIN_PROJECT=

NEO4J_AURA=
NEO4J_AURA_USERNAME=
NEO4J_AURA_PASSWORD=
NEO4J_AURA_DATABASE=

INVESTIGATION_API_KEY=
```

Neo4j and the vector index connect lazily on first use rather than at
import time, so `uv run -m src.run -- "..."` will start even with these
unset -- it just fails with a clear error the moment a log/Cypher question
actually needs the graph.

Setup is completed, now you can run the program!!!
```bash
  uv run -m src.run -- "Your question here"
```
This is the original single-question CLI: one manually-typed question, one pipeline run, printed to
stdout. For the case-based, multi-turn investigation flow (ingestion + a browsable case list + a
chatroom), see "Ingestion and investigation" below instead.


## Features

- **Multi-agent workflow**: LangGraph orchestrates guardrails, hypothesis-driven question generation,
  concurrent multi-source retrieval dispatch, evidence review, synthesis, and grounding verification.

- **Guardrails**: relevance gate for on-topic questions/alerts, plus (in the investigation flow) a
  per-turn `investigation_mode` decision -- generate a full hypothesis set, ask one direct question, or
  answer straight from evidence already gathered in a prior turn, depending on what the follow-up
  actually needs.

- **Hypothesis-driven question generation**: instead of assuming one interpretation of an alert, the
  system proposes several competing hypotheses (including at least one benign one) and generates the
  sharpest discriminating question for each, tagged with which retrieval source answers it.

- **Cypher, Vector & CSKG retrieval**: structured graph queries (Neo4j), hybrid vector+keyword search
  over alert narratives, and MCP-based lookups against the public SEPSES CSKG (MITRE ATT&CK/CVE/CWE/
  CAPEC), including a schema-grounded text-to-SPARQL fallback for questions the fixed tools don't cover.
  A bounded retry recovers from transient retrieval failures (timeouts, momentary errors) without
  retrying genuinely empty results, which are treated as evidence in their own right.

- **Structured synthesizer output + grounding check**: the synthesizer returns a typed
  `SynthesizedReport` (summary, diagnosis, MITRE techniques, mitigation suggestions, confidence,
  priority -- see `src/agents/synthesizer_agent.py`) instead of free text, and a non-LLM
  `grounding_check` node (`src/agents/grounding_check.py`) verifies every cited entity actually appears
  in the retrieved context (or the case's own fact sheet), retrying synthesis once on failure.

- **Evidence-based confidence calibration**: stated confidence is capped by how many of the three
  retrieval sources actually returned substantive content, rather than the LLM's own unconstrained
  self-assessment.

- **Temporal decay weighting**: Cypher/vector retrieval results are re-ranked by recency relative to
  the query (`src/retrieval/temporal.py`, DefenGraph eq. 5), so recent log events outrank stale ones
  with similar relevance.

- **Logging & Configuration**: structured, per-agent logging (each node logs under its own
  `agent.<name>` logger, so `%(name)s` identifies which agent produced a line at a glance) and
  configuration via `.env` files.


## Ingestion and investigation

`src/ingestion/` builds the graph-construction pipeline (this didn't exist in the original AgCyRAG --
the Neo4j graph was assumed pre-built by an external no-code tool) and a live-alert simulator. Data
comes from the [AIT Alert Data Set (AIT-ADS)](https://zenodo.org/records/8263181) -- real
Wazuh/Suricata/AMiner alerts, with genuine multi-alert host correlation and native MITRE tagging on
~36% of alerts. **Ingestion only ever stores alerts into the graph -- it never triggers an
investigation.** Investigation is a separate process, always invoked by a human:

```bash
# 1. Download a scenario's raw files from Zenodo (large, not checked into git --
#    e.g. russellmitchell_wazuh.json + labels.csv), then load a sample:
uv run -m src.ingestion.ait_ads --scenario russellmitchell \
    --wazuh-json /path/to/russellmitchell_wazuh.json \
    --labels-csv /path/to/labels.csv
# writes data/alerts.jsonl + data/alerts_ground_truth.json

# 2. Ingest into Neo4j (needs NEO4J_AURA* in .env) -- run this in its own terminal;
#    it exits once the (paced, replayed) stream is exhausted
uv run -m src.ingestion.run_live
# or, for the small bundled sample instead of the full dataset:
uv run -m src.ingestion.run_live --alerts-path data/alerts_sample.jsonl

# 3. In a SEPARATE terminal, run the investigation API (needs NEO4J_AURA* + OPENAI_API_KEY in .env)
uv run -m src.investigation.run_api
```

With the investigation API running, an analyst reviews and investigates what's been ingested:

```bash
# Ranked list of recent activity cases (clustered alerts, by urgency)
curl -s "http://127.0.0.1:8000/cases?lookback_hours=24" | python3 -m json.tool

# Launch/resume the initial investigation on a case from that list
curl -s -X POST "http://127.0.0.1:8000/cases/<case_id>/investigate" | python3 -m json.tool

# Ask a follow-up in that case's chatroom -- grounded in the case and everything found so far
curl -s -X POST "http://127.0.0.1:8000/investigations/<case_id>/chat" \
  -H "Content-Type: application/json" -d '{"message": "..."}' | python3 -m json.tool

# Or investigate an explicit time range instead of a clustered case
curl -s -X POST "http://127.0.0.1:8000/investigations/time-range" \
  -H "Content-Type: application/json" -d '{"start": "...", "end": "..."}' | python3 -m json.tool
```

FastAPI's interactive docs (`http://127.0.0.1:8000/docs`) list every route, including
`GET /cases/{case_id}/attack-graph` for a reconstructed attack-chain view of a case. See
`PROTOTYPE.md` §2.3a for the full design (clustering, urgency ranking, sigma-rule MITRE enrichment,
multi-turn memory, the evidence-accumulation window).

`eval/` scores this system's structured output (MITRE techniques, escalation priority, confidence) against
the dataset's own ground truth -- no hand-authored reference answers needed:

```bash
./run_eval.sh                # or: uv run -m eval.run_eval
./run_eval.sh --llm-judge     # also score with the LLM-as-a-Judge rubric (2604.11419v1 Sec.3.11) -- slower/costlier
```

This reports MITRE technique agreement (recall, over alerts with a native MITRE tag), an escalation confusion matrix (precision/recall/F1), and calibration (does accuracy actually rise from low- to high-stated confidence). Full per-alert results land in `eval/output/eval_results.json`.


## Project Structure

```bash

multi-agents-cykg-rag/
├── src/
│   ├── agents/
│   │   ├── guardrails_agent.py         # relevance gate + investigation_mode routing
│   │   ├── question_generation_agent.py # hypothesis-driven multi-question generation
│   │   ├── cypher_agent.py              # LLM-generated Cypher (read-only guarded)
│   │   ├── vector_agent.py              # hybrid vector+keyword search
│   │   ├── mcp_rdf_agent.py             # CSKG lookups via MCP
│   │   ├── review_agent.py              # per-question hypothesis verdicts
│   │   ├── synthesizer_agent.py         # structured SynthesizedReport
│   │   └── grounding_check.py           # non-LLM citation verification
│   ├── config/
│   │   └── settings.py                  # lazy Neo4j/LLM init, read-only Cypher guard
│   ├── graph/
│   │   ├── state.py                     # AgentState (LangGraph shared state)
│   │   └── workflow.py                  # graph wiring of all agents above
│   ├── ingestion/                       # process 1: store alerts, never investigate
│   │   ├── schema.py                    # Alert pydantic model
│   │   ├── ait_ads.py                   # AIT-ADS dataset loader
│   │   ├── graph_loader.py              # raw alert -> Neo4j
│   │   ├── live_stream.py               # paced async alert replay
│   │   ├── trigger.py                   # ingest_stream() -- ingestion only
│   │   └── run_live.py                  # CLI entrypoint
│   ├── investigation/                   # process 2: human-invoked
│   │   ├── clustering.py                # activity-case clustering + urgency ranking
│   │   ├── sigma_matcher.py             # Sigma-rule secondary MITRE tagging
│   │   ├── sigma_rules/                 # curated Sigma YAML rules
│   │   ├── attack_graph.py              # attack-chain reconstruction + root cause
│   │   ├── narrative.py                 # case_context fact sheet
│   │   ├── session.py                   # checkpointed multi-turn investigation runner
│   │   ├── api.py                       # FastAPI service
│   │   └── run_api.py                   # CLI entrypoint
│   ├── retrieval/
│   │   └── temporal.py                  # recency-based re-ranking
│   ├── utils/
│   │   └── logging_config.py
│   └── run.py                           # original single-question CLI
├── eval/                                # ground-truth-based quantitative evaluation
├── mcp-cskg-rdf/                        # bundled MCP server (SEPSES CSKG)
├── PROTOTYPE.md                         # full architecture, design rationale, limitations
├── SECURITY_ASSESSMENT.md               # threat-model self-assessment + fixes
└── .env

```
