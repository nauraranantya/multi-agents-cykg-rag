#!/bin/bash
# Runs the eval (AIT-ADS alert sample, already ingested into Neo4j and written
# to data/alerts.jsonl / data/alerts_ground_truth.json by src/ingestion/ait_ads.py)
# -- MITRE technique agreement (only over alerts with a native MITRE tag),
# escalation confusion matrix, and confidence calibration. Pass --llm-judge to
# also score each alert with the LLM-as-a-Judge rubric (slower/costlier: one
# extra LLM call per alert).
#
# Needs OPENAI_API_KEY + NEO4J_AURA* already set in multi-agents-cykg-rag/.env.
# Expect on the order of 30-45 min for the full sample through the multi-agent
# pipeline. Runs in your own terminal so you can watch it live / Ctrl+C it
# yourself.

set -e
cd "$(dirname "$0")"

rm -f eval/output/eval_results.json

.venv/bin/python -u -m eval.run_eval "$@"

echo ""
echo "Done. Full per-alert results: eval/output/eval_results.json"
