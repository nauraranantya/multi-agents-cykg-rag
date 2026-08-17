# eval/analyze_trigger_threshold.py
"""
Sweeps src/ingestion/trigger.py's TRIGGER_SEVERITY_THRESHOLD against the
AIT-ADS sample's rule_level and ground-truth should_escalate (attack-phase
window membership) to see how well raw Wazuh severity alone separates
genuine attack-phase alerts from background noise -- no LLM involved, this
is purely about whether the *gate* is well-calibrated before anything ever
reaches AgCyRAG.

Beyond the raw precision/recall table, this also does two things a plain
sweep doesn't:

  1. Pareto-dominance check -- a threshold is "dominated" if some other
     threshold has >= precision AND >= recall (with at least one strictly
     better). Dominated thresholds are never a good choice regardless of how
     you weigh precision vs. recall, so they're flagged rather than left for
     the reader to notice by eye.
  2. F-beta scoring at beta=1 (equal weight) and beta=2 (recall weighted 4x
     as important as precision -- a rough proxy for security alerting's
     usual cost asymmetry: a missed attack is normally far more expensive
     than an analyst spending a few minutes on a false alarm). The
     "best F1" and "best F2" thresholds are reported separately since they
     can genuinely differ, and which one is "right" is a security-policy
     question, not a statistical one -- this script computes both rather
     than picking for you.

Reads local data/alerts.jsonl + data/alerts_ground_truth.json. No API keys
needed.

Usage:
    uv run -m eval.analyze_trigger_threshold
"""
from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.trigger import TRIGGER_SEVERITY_THRESHOLD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_records() -> list[dict]:
    gt = json.load(open(DATA_DIR / "alerts_ground_truth.json"))
    records = []
    with open(DATA_DIR / "alerts.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            g = gt.get(a["id"])
            if g is None:
                continue
            records.append({
                "id": a["id"],
                "rule_level": a["rule_level"],
                "rule_description": a["rule_description"],
                "should_escalate": g["should_escalate"],
                "phase": g.get("phase"),
            })
    return records


def f_beta(precision: float, recall: float, beta: float) -> float:
    if precision != precision or recall != recall:  # nan check without importing math
        return float("nan")
    num = (1 + beta ** 2) * precision * recall
    denom = (beta ** 2 * precision) + recall
    return num / denom if denom else 0.0


def sweep(records: list[dict]):
    n_genuine = sum(1 for r in records if r["should_escalate"])
    n_noise = sum(1 for r in records if not r["should_escalate"])
    print(f"Sample: {len(records)} alerts ({n_genuine} genuine/attack-phase, {n_noise} noise)\n")

    # Only evaluate thresholds where the triggered set actually changes --
    # every distinct level present, plus one past the max (nothing left
    # triggers). Sweeping every integer produces long runs of identical rows
    # whenever a level is unused (e.g. no alerts sit at exactly level 4 or 9
    # in typical Wazuh output), which just obscures the real decision points.
    levels = sorted(set(r["rule_level"] for r in records))
    # Always include the current default explicitly, even if it falls
    # between two real levels (e.g. default=7 but nothing sits at exactly
    # 7/8/9) -- otherwise it silently gets absorbed into whichever
    # neighboring level happens to produce the same triggered set, and the
    # "current default" marker below would never appear anywhere.
    candidate_thresholds = sorted(set(levels + [levels[-1] + 1, TRIGGER_SEVERITY_THRESHOLD]))

    header = (
        f"{'thresh':>6} | {'triggered':>9} | {'genuine caught':>14} | {'genuine missed':>14} | "
        f"{'noise triggered':>16} | {'precision':>9} | {'recall':>7} | {'F1':>6} | {'F2':>6}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for t in candidate_thresholds:
        triggered = [r for r in records if r["rule_level"] >= t]
        genuine_caught = sum(1 for r in triggered if r["should_escalate"])
        genuine_missed = n_genuine - genuine_caught
        noise_triggered = sum(1 for r in triggered if not r["should_escalate"])
        precision = genuine_caught / len(triggered) if triggered else float("nan")
        recall = genuine_caught / n_genuine if n_genuine else float("nan")
        rows.append({
            "t": t, "n_triggered": len(triggered), "genuine_caught": genuine_caught,
            "genuine_missed": genuine_missed, "noise_triggered": noise_triggered,
            "precision": precision, "recall": recall,
            "f1": f_beta(precision, recall, 1.0), "f2": f_beta(precision, recall, 2.0),
        })

    # Pareto dominance: t is dominated if some other t' has precision>= and
    # recall>= with at least one strictly greater.
    def is_dominated(row, others):
        for o in others:
            if o["t"] == row["t"]:
                continue
            not_worse = o["precision"] >= row["precision"] and o["recall"] >= row["recall"]
            strictly_better = o["precision"] > row["precision"] or o["recall"] > row["recall"]
            if not_worse and strictly_better:
                return True
        return False

    for row in rows:
        row["dominated"] = is_dominated(row, rows)

    for row in rows:
        marker = ""
        if row["t"] == TRIGGER_SEVERITY_THRESHOLD:
            marker += "  <- current default"
        if row["dominated"]:
            marker += "  [DOMINATED -- some other threshold is >= on both precision and recall]"
        print(
            f"{row['t']:>6} | {row['n_triggered']:>9} | {row['genuine_caught']:>14} | "
            f"{row['genuine_missed']:>14} | {row['noise_triggered']:>16} | "
            f"{row['precision']:>9.3f} | {row['recall']:>7.3f} | {row['f1']:>6.3f} | {row['f2']:>6.3f}{marker}"
        )

    scorable = [r for r in rows if r["precision"] == r["precision"]]  # drop nan rows
    if scorable:
        best_f1 = max(scorable, key=lambda r: r["f1"])
        best_f2 = max(scorable, key=lambda r: r["f2"])
        print()
        print(
            f"Best F1 (precision/recall weighted equally): threshold={best_f1['t']} "
            f"(precision={best_f1['precision']:.3f}, recall={best_f1['recall']:.3f}, F1={best_f1['f1']:.3f})"
        )
        print(
            f"Best F2 (recall weighted 4x precision -- rough security-alerting cost proxy): "
            f"threshold={best_f2['t']} "
            f"(precision={best_f2['precision']:.3f}, recall={best_f2['recall']:.3f}, F2={best_f2['f2']:.3f})"
        )
        if best_f1["t"] != best_f2["t"]:
            print(
                "These disagree -- which one is 'right' depends on how costly a missed attack is "
                "relative to an analyst reviewing a false alarm, not something this script can decide."
            )

    print()
    threshold = TRIGGER_SEVERITY_THRESHOLD
    missed = [r for r in records if r["should_escalate"] and r["rule_level"] < threshold]
    if missed:
        print(f"Genuine attack-phase alerts MISSED at threshold={threshold} (never reach AgCyRAG in production):")
        for r in missed:
            print(f"  - {r['id']} | level={r['rule_level']} | phase={r['phase']} | {r['rule_description']!r}")
    else:
        print(f"No genuine attack-phase alerts missed at threshold={threshold} in this sample.")

    noise_triggered = [r for r in records if not r["should_escalate"] and r["rule_level"] >= threshold]
    print(f"\nNoise alerts that WOULD still trigger at threshold={threshold}: {len(noise_triggered)}")
    for r in noise_triggered[:10]:
        print(f"  - {r['id']} | level={r['rule_level']} | {r['rule_description']!r}")
    if len(noise_triggered) > 10:
        print(f"  ... and {len(noise_triggered) - 10} more")


if __name__ == "__main__":
    sweep(load_records())
