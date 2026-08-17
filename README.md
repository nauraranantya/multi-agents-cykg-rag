
# AgCyRAG: an Agentic Knowledge Graph based RAG Framework for Automated Security Analysis

AgCyRAG is a hybrid Agentic Retrieval-Augmented Generation (RAG) framework designed to improve cybersecurity analysis by integrating Knowledge Graph (KG) reasoning with vector-based retrieval.
It enables factual grounding of Large Language Model (LLM)-powered analyses while handling heterogeneous structured and unstructured data (e.g., security log sources)
## Core Components

- Multi-Agent System (LangGraph): The primary application logic that orchestrates the entire workflow. It includes specialized agents for validating questions, querying databases, reflecting on results, and synthesizing final answers
- Neo4j Knowledge Graph: A graph database storing structured cybersecurity data (e.g., from the MITRE ATT&CK framework), which is queried using the Cypher language.
- MCP RDF Explorer: Model Context Protocol (MCP) server that provides a conversational interface for RDF-based Knowledge Graph (Turtle) exploration and analysis in local file mode or SPARQL endpoint mode. (https://github.com/sepses/multi-agents-cykg-rag/tree/main/mcp-cskg-rdf)

## Setup and Installation

How to use

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
  accept it fine) -- `server.py` now injects `truststore` at startup (verifies
  against the OS trust store instead) specifically to handle this.

Make sure again that the .env file is filled !!!

`src/config/settings.py` only ever reads the `NEO4J_AURA*` variables below (a
managed Neo4j AuraDB instance) -- there is no local/self-hosted Neo4j path in
the code, regardless of what earlier versions of this README implied.
`LANGCHAIN_*` are optional (LangSmith tracing); leave them blank to skip
tracing.

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
```

Neo4j and the vector index now connect lazily on first use rather than at
import time, so `uv run -m src.run -- "..."` will start even with these
unset -- it just fails with a clear error the moment a log/Cypher question
actually needs the graph.

Setup is completed, now you can run the program!!!
```bash
  uv run -m src.run -- "Your question here"
```


## Features

- **Multi-agent workflow**: Uses LangGraph to manage the workflow between agents (guardrails, vector search, cypher search, MCP RDF, reflection, and synthesizer).

- **Guardrails**: Ensures the questions asked are relevant to the cybersecurity domain.

- **Vector & Cypher Search**: Searches for answers from a vector and graph database (Neo4j) with automatic iteration and reflection if results are insufficient.

- **MCP RDF Agent**: Integrates RDF-based search to enrich the context of answers.

- **Synthesizer**: Combines results from multiple sources into a comprehensive final answer.

- **Logging & Configuration**: Supports structured logging and configuration via .env files.

- **Structured synthesizer output + grounding check**: the synthesizer returns a typed `SynthesizedReport` (see `src/agents/synthesizer_agent.py`) instead of free text, and a non-LLM `grounding_check` node (`src/agents/grounding_check.py`) verifies every cited entity actually appears in the retrieved context, retrying synthesis once on failure.

- **Temporal decay weighting**: Cypher/vector retrieval results are re-ranked by recency relative to the query (`src/retrieval/temporal.py`, DefenGraph eq. 5), so recent log events outrank stale ones with similar relevance.

## Prototype: log ingestion, auto-triggering, and evaluation

`src/ingestion/` adds the graph-construction pipeline that didn't exist before (see README's Setup section above) plus a live-alert simulator and an auto-triggering wrapper around the pipeline above. Data comes from the [AIT Alert Data Set (AIT-ADS)](https://zenodo.org/records/8263181) -- real Wazuh/Suricata/AMiner alerts from the same AIT research lineage as this project's own cited log dataset, with genuine multi-alert host correlation and native MITRE tagging on ~36% of alerts:

```bash
# 1. Download a scenario's raw files from Zenodo (large, not checked into git --
#    e.g. russellmitchell_wazuh.json + labels.csv), then load a sample:
uv run -m src.ingestion.ait_ads --scenario russellmitchell \
    --wazuh-json /path/to/russellmitchell_wazuh.json \
    --labels-csv /path/to/labels.csv
# writes data/alerts.jsonl + data/alerts_ground_truth.json

# 2. Ingest into Neo4j (needs NEO4J_AURA* in .env)
uv run -m src.ingestion.graph_loader

# 3. Replay the alert stream live and auto-trigger AgCyRAG for severity-gated alerts
#    (needs NEO4J_AURA* + OPENAI_API_KEY in .env)
uv run -m src.ingestion.run_live
```

Every alert is ingested into the graph regardless of severity; only alerts at or above `TRIGGER_SEVERITY_THRESHOLD` (env var, default `7`) fire the full multi-agent pipeline. Results land in `data/auto_trigger_log.jsonl`. Note: on real AIT-ADS data this threshold currently lets almost nothing through -- see `eval/analyze_trigger_threshold.py` for why (Wazuh's default ruleset assigns most base-rule alerts a flat low severity regardless of whether they're part of an active attack).

`eval/` scores AgCyRAG's structured output (MITRE techniques, escalation priority, confidence) against the dataset's own ground truth -- no hand-authored reference answers needed:

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
│   │   ├── __init__.py
│   │   ├── guardrails_agent.py
│   │   ├── mcp_rdf_agent.py
│   │   ├── cypher_agent.py
│   │   ├── log_analysis_agent.py
│   │   ├── reflection_agent.py
│   │   ├── review_agent.py
│   │   ├── routing_agent.py (dead code, not wired into workflow.py)
│   │   ├── synthesizer_agent.py
│   │   └── vector_agent.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── workflow.py
│   ├── log/
│   │   └── (a log file will be created here)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logging_config.py
│   └── run.py
└── .env


```

