# eval/run_eval.py
"""
Evaluates AgCyRAG against the AIT-ADS alert sample loaded into
data/alerts.jsonl + data/alerts_ground_truth.json (see
src/ingestion/ait_ads.py). Reads local files, no API calls needed for
loading -- adapted for two AIT-ADS-specific realities:

  - Not every alert has a native MITRE tag (only ~36% of raw AIT-ADS alerts
    do, per src/ingestion/ait_ads.py's own count). MITRE agreement is only
    computed over the subset that has one (`has_mitre_ground_truth`) --
    scoring "no tag present" as a recall miss against nothing would be
    wrong, not a real result.
  - should_escalate is derived from labeled attack-phase time windows
    (labels.csv), not a native per-alert field -- already baked into
    alerts_ground_truth.json by ait_ads.py, nothing extra needed here.

Needs real OPENAI_API_KEY / NEO4J_AURA* credentials.

Usage:
    uv run -m eval.run_eval
    uv run -m eval.run_eval --alerts-path data/alerts.jsonl --ground-truth-path data/alerts_ground_truth.json
    uv run -m eval.run_eval --llm-judge     # also score with the LLM-as-a-Judge rubric (slower/costlier)
    uv run -m eval.run_eval --gate-filter   # only run alerts that pass TRIGGER_SEVERITY_THRESHOLD,
                                             # report end-to-end (gate + LLM) precision/recall
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List

from src.graph.workflow import app
from src.ingestion.graph_loader import ingest_alert
from src.ingestion.schema import Alert
from src.ingestion.trigger import TRIGGER_SEVERITY_THRESHOLD, _trigger_reason
from eval.metrics import (
    PRIORITY_TO_ESCALATE,
    calibration_by_confidence,
    confusion_counts,
    precision_recall_f1,
    score_escalation_agreement,
    score_llm_judge,
    score_mitre_agreement,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "output"


def load_alerts_with_ground_truth(alerts_path: Path, ground_truth_path: Path) -> List[dict]:
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    records = []
    with open(alerts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            alert = Alert(**json.loads(line))
            gt = ground_truth.get(alert.id)
            if gt is None:
                continue  # alerts.jsonl has an entry with no matching ground-truth record -- skip, don't guess
            records.append({"alert": alert, "ground_truth": gt})
    return records


def build_baseline_answer(gt: dict) -> str:
    """This dataset has no human-authored narrative reference answer --
    ground truth is structural (attack-phase window membership, native MITRE
    tag), same as the rest of this eval. score_llm_judge (eval/metrics.py)
    needs *some* baseline text to compare against though, so this synthesizes
    the minimal factual statement the ground truth actually supports -- not a
    full analyst writeup, just the facts a correct answer must not
    contradict. Judge scoring is about agreement/faithfulness against these
    facts, not prose quality of the baseline itself."""
    if gt["should_escalate"]:
        parts = [f"This alert occurred during the '{gt.get('phase')}' attack phase and should be escalated."]
    else:
        parts = ["This alert is background/noise activity, not part of any labeled attack phase, and should not be escalated."]
    if gt.get("has_native_mitre_tag"):
        parts.append(f"The associated MITRE ATT&CK technique is {gt.get('mitre_technique')}.")
    else:
        parts.append("No MITRE ATT&CK technique is recorded for this alert.")
    return " ".join(parts)


def build_question(alert: Alert) -> str:
    parts = [f"A security alert was raised on host '{alert.agent_name}': {alert.rule_description}"]
    if alert.data_srcip:
        parts.append(f"source IP {alert.data_srcip}")
    parts.append(
        "Investigate what happened, identify any related MITRE ATT&CK techniques, "
        "and recommend a response"
    )
    return ". ".join(parts) + "."


async def run_one(record: dict, use_llm_judge: bool = False, gate_filter: bool = False) -> dict:
    alert: Alert = record["alert"]
    gt = record["ground_truth"]

    ingest_alert(alert)  # idempotent -- safe even if already ingested via graph_loader.load_all()
    # Ingest unconditionally (mirrors trigger.py's real behavior: every alert
    # joins the graph so other alerts have it to correlate against) even when
    # the gate below skips running the pipeline on THIS alert.

    trigger_reason = _trigger_reason(alert)  # src/ingestion/trigger.py -- severity OR native MITRE tag
    gate_triggered = trigger_reason is not None
    if gate_filter and not gate_triggered:
        # Matches production: trigger.py never invokes the pipeline for
        # alerts that trigger neither condition. No LLM call, no cost --
        # this alert only ever shows up as a ground-truth denominator for
        # end-to-end recall (main()'s summary), never as a scored prediction.
        return {
            "alert_id": alert.id,
            "scenario": gt.get("scenario"),
            "true_phase": gt.get("phase"),
            "question": None,
            "answer": None,
            "error": None,
            "grounded": None,
            "confidence": None,
            "predicted_techniques": [],
            "true_technique": gt.get("mitre_technique"),
            "has_mitre_ground_truth": bool(gt.get("has_native_mitre_tag")),
            "technique_hit": None,
            "predicted_priority": None,
            "true_should_escalate": gt["should_escalate"],
            "escalation_correct": None,
            "latency_seconds": 0.0,
            "llm_judge": None,
            "gate_triggered": False,
            "trigger_reason": None,
        }

    question = build_question(alert)
    initial_state = {
        "question": question,
        "original_question": question,
        "messages": [("human", question)],
        "cypher_iteration_count": 1,
        "vector_iteration_count": 1,
        "max_iterations": 3,
        "query_timestamp": alert.timestamp,
    }

    start = time.monotonic()
    error = None
    result = {}
    try:
        result = await app.ainvoke(initial_state, config={"recursion_limit": 30})
    except Exception as e:
        error = str(e)
    latency = time.monotonic() - start

    report = result.get("synthesized_report") or {}
    grounding = result.get("grounding_result") or {}
    predicted_techniques = report.get("mitre_techniques", [])
    predicted_priority = report.get("recommended_priority")
    confidence = report.get("confidence")

    has_mitre_gt = bool(gt.get("has_native_mitre_tag"))
    mitre = score_mitre_agreement(predicted_techniques, gt.get("mitre_technique")) if has_mitre_gt else None
    escalation = score_escalation_agreement(predicted_priority, gt["should_escalate"])

    judge = None
    if use_llm_judge and not error:
        try:
            judge = score_llm_judge(question, build_baseline_answer(gt), result.get("answer"))
        except Exception as e:
            judge = {"error": str(e)}

    return {
        "alert_id": alert.id,
        "scenario": gt.get("scenario"),
        "true_phase": gt.get("phase"),
        "question": question,
        "answer": result.get("answer"),
        "error": error,
        "grounded": grounding.get("grounded"),
        "confidence": confidence,
        "predicted_techniques": predicted_techniques,
        "true_technique": gt.get("mitre_technique"),
        "has_mitre_ground_truth": has_mitre_gt,
        "technique_hit": mitre.technique_hit if mitre else None,
        "predicted_priority": predicted_priority,
        "true_should_escalate": gt["should_escalate"],
        "escalation_correct": escalation.correct,
        "latency_seconds": round(latency, 2),
        "llm_judge": judge,
        "gate_triggered": gate_triggered,
        "trigger_reason": trigger_reason,
    }


async def main(alerts_path: Path, ground_truth_path: Path, use_llm_judge: bool = False, gate_filter: bool = False):
    records = load_alerts_with_ground_truth(alerts_path, ground_truth_path)
    print(f"Loaded {len(records)} alerts with ground truth from {alerts_path.name}.")
    if use_llm_judge:
        print("LLM-as-a-Judge scoring enabled -- one extra LLM call per alert (2604.11419v1 Sec.3.11 rubric).")
    if gate_filter:
        print(
            f"Gate filter enabled -- only alerts with rule_level >= {TRIGGER_SEVERITY_THRESHOLD} "
            f"(TRIGGER_SEVERITY_THRESHOLD) OR a native MITRE ATT&CK tag are run through the pipeline, "
            f"matching trigger.py's real production behavior. Alerts triggering neither are skipped "
            f"(no LLM call) and count only toward end-to-end recall's denominator below."
        )

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "eval_results.json"

    results = []
    for i, record in enumerate(records):
        alert = record["alert"]
        gt = record["ground_truth"]
        print(
            f"--- [{i + 1}/{len(records)}] {alert.id} "
            f"(level={alert.rule_level}, true_phase={gt.get('phase')}, should_escalate={gt['should_escalate']}) ---"
        )
        r = await run_one(record, use_llm_judge=use_llm_judge, gate_filter=gate_filter)
        results.append(r)
        if gate_filter and not r["gate_triggered"]:
            status = "SKIPPED (below trigger threshold)"
        elif r["error"]:
            status = "ERROR"
        else:
            status = (
                f"technique_hit={r['technique_hit']} (gt={'yes' if r['has_mitre_ground_truth'] else 'n/a'}) "
                f"escalation_correct={r['escalation_correct']}"
            )
            if r.get("llm_judge") and "weighted_total" in r["llm_judge"]:
                status += f" judge={r['llm_judge']['weighted_total']}/50"
        print(f"    {status}")

        # Checkpoint after every alert so a partial run (killed, crashed, or
        # still in progress) always has a readable results file on disk --
        # previously this only got written once, after the full loop finished.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # "scored" = actually ran through the pipeline. Under --gate-filter,
    # excludes gate-skipped alerts (error=None but never invoked -- they'd
    # otherwise silently pollute the LLM-quality metrics below with
    # predicted_priority=None entries that were never really "wrong", just
    # never asked). Without --gate-filter, every alert was run regardless of
    # level, so this is unchanged from before.
    scored = [r for r in results if not r["error"] and (not gate_filter or r["gate_triggered"])]
    mitre_scorable = [r for r in scored if r["has_mitre_ground_truth"]]
    n_technique_hit = sum(1 for r in mitre_scorable if r["technique_hit"])

    escalation_records = [score_escalation_agreement(r["predicted_priority"], r["true_should_escalate"]) for r in scored]
    e_counts = confusion_counts(escalation_records)
    e_prf = precision_recall_f1(e_counts)
    calibration = calibration_by_confidence(
        [{"confidence": r["confidence"], "correct": r["escalation_correct"]} for r in scored]
    )

    n_gate_skipped = sum(1 for r in results if gate_filter and not r["gate_triggered"])
    n_errored = sum(1 for r in results if r["error"])
    print("=" * 70)
    print(
        f"{len(results)} alerts loaded, {len(scored)} scored, "
        f"{n_gate_skipped} below gate threshold (skipped, not errors), {n_errored} errored"
    )
    if mitre_scorable:
        print(
            f"MITRE technique agreement (recall, over {len(mitre_scorable)}/{len(scored)} alerts "
            f"with a native MITRE tag): {n_technique_hit}/{len(mitre_scorable)} = "
            f"{n_technique_hit / len(mitre_scorable):.3f}"
        )
    else:
        print("MITRE technique agreement: no alerts in this sample had a native MITRE tag to score against.")
    print(f"Escalation confusion matrix: {e_counts}")
    print(f"Escalation precision/recall/F1: {e_prf}")
    print(f"Calibration (accuracy by stated confidence -- should rise low->high): {calibration}")

    if gate_filter:
        # End-to-end (gate + LLM) precision/recall, decomposed into the two
        # stages -- distinct from the confusion matrix above, which only
        # covers alerts that actually reached the LLM (scored). This answers
        # "how well does the deployed system perform", not "how good is the
        # LLM's judgment given that it was asked".
        true_attacks = [r for r in results if r["true_should_escalate"]]
        true_attacks_triggered = [r for r in true_attacks if r["gate_triggered"]]
        true_attacks_escalated = [
            r for r in true_attacks_triggered
            if not r["error"] and PRIORITY_TO_ESCALATE.get((r["predicted_priority"] or "").lower()) is True
        ]
        all_flagged_escalate = [
            r for r in results
            if r["gate_triggered"] and not r["error"]
            and PRIORITY_TO_ESCALATE.get((r["predicted_priority"] or "").lower()) is True
        ]
        flagged_correct = [r for r in all_flagged_escalate if r["true_should_escalate"]]

        gate_recall = len(true_attacks_triggered) / len(true_attacks) if true_attacks else float("nan")
        llm_recall_given_triggered = (
            len(true_attacks_escalated) / len(true_attacks_triggered) if true_attacks_triggered else float("nan")
        )
        e2e_recall = len(true_attacks_escalated) / len(true_attacks) if true_attacks else float("nan")
        e2e_precision = len(flagged_correct) / len(all_flagged_escalate) if all_flagged_escalate else float("nan")

        n_by_reason = {"severity": 0, "mitre_tag": 0}
        for r in true_attacks_triggered:
            n_by_reason[r["trigger_reason"]] += 1

        print("-" * 70)
        print(f"End-to-end system (gate + LLM) against all {len(true_attacks)} true attack-phase alerts in this sample:")
        print(
            f"  Gate recall (rule_level >= {TRIGGER_SEVERITY_THRESHOLD} OR native MITRE tag reaches the LLM at all): "
            f"{len(true_attacks_triggered)}/{len(true_attacks)} = {gate_recall:.3f} "
            f"({n_by_reason['severity']} via severity, {n_by_reason['mitre_tag']} via MITRE tag alone)"
        )
        print(
            f"  LLM recall, conditional on reaching the gate: "
            f"{len(true_attacks_escalated)}/{len(true_attacks_triggered)} = {llm_recall_given_triggered:.3f}"
            if true_attacks_triggered else "  LLM recall, conditional on reaching the gate: n/a (no true attacks reached the gate)"
        )
        print(f"  End-to-end recall (gate x LLM): {len(true_attacks_escalated)}/{len(true_attacks)} = {e2e_recall:.3f}")
        print(
            f"  End-to-end precision (of everything the system actually flagged escalate): "
            f"{len(flagged_correct)}/{len(all_flagged_escalate)} = {e2e_precision:.3f}"
            if all_flagged_escalate else "  End-to-end precision: n/a (system flagged nothing as escalate)"
        )

    judge_scored = [r["llm_judge"] for r in scored if r.get("llm_judge") and "weighted_total" in r["llm_judge"]]
    if judge_scored:
        n = len(judge_scored)
        mean_total = sum(j["weighted_total"] for j in judge_scored) / n
        mean_agreement = sum(j["agreement"] for j in judge_scored) / n
        mean_adequacy = sum(j["adequacy"] for j in judge_scored) / n
        mean_faithfulness = sum(j["faithfulness"] for j in judge_scored) / n
        mean_clarity = sum(j["clarity"] for j in judge_scored) / n
        print(
            f"LLM-as-a-Judge (n={n}): mean weighted total {mean_total:.1f}/50 "
            f"(agreement={mean_agreement:.2f}/5, adequacy={mean_adequacy:.2f}/5, "
            f"faithfulness={mean_faithfulness:.2f}/5, clarity={mean_clarity:.2f}/5)"
        )
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AgCyRAG against a local alert sample.")
    parser.add_argument("--alerts-path", default=str(DATA_DIR / "alerts.jsonl"))
    parser.add_argument("--ground-truth-path", default=str(DATA_DIR / "alerts_ground_truth.json"))
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Also score each alert with the LLM-as-a-Judge rubric (2604.11419v1 Sec.3.11). "
             "One extra LLM call per alert -- off by default to keep runs fast/cheap.",
    )
    parser.add_argument(
        "--gate-filter", action="store_true",
        help="Only run the pipeline on alerts that pass TRIGGER_SEVERITY_THRESHOLD "
             "(src/ingestion/trigger.py), matching real production behavior. Reports "
             "end-to-end (gate + LLM) precision/recall against ALL true attack-phase alerts "
             "in the sample, not just the ones that reached the LLM. Saves cost too, since "
             "below-threshold alerts skip the LLM call entirely.",
    )
    args = parser.parse_args()
    asyncio.run(main(
        Path(args.alerts_path), Path(args.ground_truth_path),
        use_llm_judge=args.llm_judge, gate_filter=args.gate_filter,
    ))
