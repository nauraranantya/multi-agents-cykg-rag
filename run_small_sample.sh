#!/bin/bash
# Smoke-tests the full live-ingestion pipeline (trigger.py, now wired to
# alert_triage_agent.py -- see trigger.py's ENABLE_LLM_TRIAGE docstring) end
# to end, against a small slice of the already-sampled AIT-ADS data
# (data/alerts.jsonl) instead of the full ~178-alert set. Useful after
# changing trigger.py/question_generation_agent.py/etc. to see it actually
# run before committing to a full eval.
#
# What it does:
#   1. Takes the first N alerts from data/alerts.jsonl (order doesn't matter
#      here -- live_stream.py re-sorts by timestamp) into data/alerts_sample.jsonl.
#   2. Replays just that sample through src.ingestion.run_live: every alert
#      is ingested into Neo4j unconditionally; alerts passing the trigger
#      gate (severity >= threshold, OR a native MITRE tag, OR now the LLM
#      triage classifier -- see trigger.py) run the full multi-agent
#      pipeline.
#
# Cost note: ENABLE_LLM_TRIAGE defaults to "true", so every sampled alert
# that the two deterministic signals would otherwise skip now costs one
# extra LLM call for the triage check, on top of whatever alerts actually
# get fully investigated. Set ENABLE_LLM_TRIAGE=false before running this
# to skip that and test the deterministic-only gate instead.
#
# Needs OPENAI_API_KEY + NEO4J_AURA* already set in multi-agents-cykg-rag/.env
# -- this writes real alerts into your live Neo4j graph and makes real LLM
# calls, same as run_live.py normally does. Runs in your own terminal so you
# can watch it live / Ctrl+C it yourself.
#
# Usage:
#   ./run_small_sample.sh            # 8 alerts (default)
#   ./run_small_sample.sh 15         # 15 alerts
#   ENABLE_LLM_TRIAGE=false ./run_small_sample.sh

set -e
cd "$(dirname "$0")"

N="${1:-8}"
SRC="data/alerts.jsonl"
SAMPLE="data/alerts_sample.jsonl"

if [ ! -f "$SRC" ]; then
    echo "Missing $SRC -- run src.ingestion.ait_ads first (see README) to produce it."
    exit 1
fi

head -n "$N" "$SRC" > "$SAMPLE"
echo "Wrote $(wc -l < "$SAMPLE" | tr -d ' ') alert(s) to $SAMPLE (from $SRC)."
echo "ENABLE_LLM_TRIAGE=${ENABLE_LLM_TRIAGE:-true} (unset means the default, true)"
echo ""

.venv/bin/python -u -m src.ingestion.run_live \
    --alerts-path "$SAMPLE" \
    --speed 50000 \
    --max-gap 0.1 \
    --max-concurrent-investigations 2

echo ""
echo "Done. Per-alert results appended to data/auto_trigger_log.jsonl."
