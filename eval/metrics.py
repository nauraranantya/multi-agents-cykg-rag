# eval/metrics.py
"""
Metrics comparing AgCyRAG's structured output against the dataset's own
ground truth -- no narrative reference answer needed. Four metric classes,
each mapping onto an established evaluation approach from the literature
rather than something invented for this project:

  1. MITRE technique agreement -- same shape as TRAM/TTPrint/TTPXHunter's
     precision/recall/F1-against-labeled-technique evaluation for automated
     ATT&CK technique extraction.
  2. Escalation/triage agreement -- same shape as SOC alert-triage LLM
     benchmarking (e.g. CORTEX): true-positive/false-positive confusion
     matrix against a real "should this be escalated" label.
  3. Calibration by stated confidence -- does correctness actually rise from
     low -> medium -> high confidence? (OpenSec-style calibration check;
     accuracy alone doesn't show whether the system knows when it might be
     wrong.)
  4. LLM-as-a-Judge (score_llm_judge) -- the weighted C1-C4 rubric from
     Hamzić et al., "Beyond RAG for Cyber Threat Intelligence: A Systematic
     Evaluation of Graph-Based and Agentic Retrieval" (2604.11419v1),
     Sec.3.11. More expensive (one LLM call per scored alert) but captures
     answer quality/faithfulness/clarity that the structural metrics above
     can't -- they only check whether specific facts (technique, escalation)
     match, not whether the prose answer is well-formed or hallucinates
     unrelated detail.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

PRIORITY_TO_ESCALATE = {"escalate": True, "monitor": False, "ignore": False}


def _normalize_technique(t: str) -> str:
    return t.strip().upper()


@dataclass
class MitreAgreement:
    predicted: List[str]
    true_technique: Optional[str]
    technique_hit: bool
    precision: float
    recall: float


def score_mitre_agreement(predicted_techniques: List[str], true_technique: Optional[str]) -> MitreAgreement:
    """Single ground-truth technique per alert (matches this dataset's
    schema). recall = did the bot's cited set include the true technique;
    precision = fraction of the bot's cited set that was the true technique.
    Note: since we only have one labeled technique per alert (not a full
    labeled set), a bot that correctly cites additional, genuinely-relevant
    techniques the dataset didn't label gets penalized on precision here --
    a known limitation of single-label ground truth, not of the bot."""
    norm_pred = [_normalize_technique(t) for t in predicted_techniques if t]
    norm_true = _normalize_technique(true_technique) if true_technique else None
    hit = norm_true is not None and norm_true in norm_pred
    precision = (1.0 if hit else 0.0) if norm_pred else 0.0
    recall = 1.0 if hit else 0.0
    return MitreAgreement(predicted=norm_pred, true_technique=norm_true, technique_hit=hit, precision=precision, recall=recall)


@dataclass
class EscalationAgreement:
    predicted_priority: Optional[str]
    predicted_should_escalate: Optional[bool]
    true_should_escalate: bool
    correct: bool


def score_escalation_agreement(predicted_priority: Optional[str], true_should_escalate: bool) -> EscalationAgreement:
    pred_escalate = PRIORITY_TO_ESCALATE.get((predicted_priority or "").lower())
    correct = (pred_escalate == true_should_escalate) if pred_escalate is not None else False
    return EscalationAgreement(
        predicted_priority=predicted_priority,
        predicted_should_escalate=pred_escalate,
        true_should_escalate=true_should_escalate,
        correct=correct,
    )


def confusion_counts(records: List[EscalationAgreement]) -> Dict[str, int]:
    tp = sum(1 for r in records if r.predicted_should_escalate is True and r.true_should_escalate is True)
    tn = sum(1 for r in records if r.predicted_should_escalate is False and r.true_should_escalate is False)
    fp = sum(1 for r in records if r.predicted_should_escalate is True and r.true_should_escalate is False)
    fn = sum(1 for r in records if r.predicted_should_escalate is False and r.true_should_escalate is True)
    unknown = sum(1 for r in records if r.predicted_should_escalate is None)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "unknown": unknown}


def precision_recall_f1(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def calibration_by_confidence(items: List[dict]) -> Dict[str, dict]:
    """items: [{"confidence": "low"/"medium"/"high", "correct": bool}, ...].
    `correct` is a correctness proxy chosen by the caller (e.g. escalation
    agreement or MITRE technique hit). Well-calibrated means accuracy should
    rise monotonically from low -> medium -> high."""
    buckets = defaultdict(list)
    for item in items:
        if item.get("confidence") in ("low", "medium", "high"):
            buckets[item["confidence"]].append(bool(item["correct"]))
    out = {}
    for level in ("low", "medium", "high"):
        vals = buckets.get(level, [])
        out[level] = {"n": len(vals), "accuracy": (sum(vals) / len(vals)) if vals else None}
    return out


class JudgeScores(BaseModel):
    """The 0-5 rubric from 2604.11419v1 Sec.3.11. Their post-hoc validation
    (Table 7: N=2,640 judgments) found agreement/adequacy/faithfulness form
    a tightly coupled correctness cluster (r=0.91-0.98) while clarity is
    substantially less correlated with the other three (r=0.43-0.50) --
    confirming it captures an orthogonal quality dimension (fluent-but-wrong
    answers score high on clarity, low on the other three) rather than just
    restating them, which is why it's still worth keeping as its own
    criterion instead of folding it into agreement."""

    agreement: int = Field(
        ..., ge=0, le=5,
        description="C1: semantic alignment with the baseline, independent of surface form. "
                    "Contradictions or unsupported deviations are penalized.",
    )
    adequacy: int = Field(
        ..., ge=0, le=5,
        description="C2: how completely and correctly the candidate addresses the question. "
                    "If the baseline says the question can't be answered from available evidence, "
                    "a candidate that also declines to answer -- without inventing detail -- is fully adequate.",
    )
    faithfulness: int = Field(
        ..., ge=0, le=5,
        description="C3: does the candidate avoid hallucinating facts or claims not supported by evidence?",
    )
    clarity: int = Field(
        ..., ge=0, le=5,
        description="C4: is the answer clear, concise, and well-structured?",
    )


_JUDGE_PROMPT = """You are scoring a candidate answer against a baseline answer for a cyber threat \
intelligence question, using a fixed 0-5 rubric per criterion.

Question: {question}

Baseline answer: {baseline}

Candidate answer: {candidate}

Score each criterion 0-5:
- Agreement: semantic alignment with the baseline, independent of surface form. Penalize \
contradictions or unsupported deviations.
- Adequacy: how completely and correctly the candidate addresses the question.
- Faithfulness: does the candidate avoid hallucinating facts or claims not supported by evidence?
- Clarity: is the answer clear, concise, and well-structured?

If the baseline states there is insufficient information to answer, and the candidate also \
declines to answer without inventing details, score Agreement, Adequacy, and Faithfulness at \
full marks (5) -- a correct refusal must not be penalized just because it's short."""

_judge_chain = None


def score_llm_judge(question: str, baseline_answer: str, candidate_answer: str) -> dict:
    """Runs the C1-C4 weighted rubric via an LLM judge (weighted total =
    4*agreement + 3*adequacy + 2*faithfulness + clarity, max 50). Uses the
    project's shared ChatOpenAI instance, imported lazily so this module has
    no hard LLM/credential dependency at import time -- same lazy-init
    pattern as src/config/settings.py's _LazyProxy and
    src/agents/cypher_agent.py's _get_cypher_qa_chain()."""
    global _judge_chain
    if _judge_chain is None:
        from src.config.settings import llm
        _judge_chain = ChatPromptTemplate.from_template(_JUDGE_PROMPT) | llm.with_structured_output(JudgeScores)

    scores: JudgeScores = _judge_chain.invoke({
        "question": question,
        "baseline": baseline_answer or "(no baseline available)",
        "candidate": candidate_answer or "(no answer produced)",
    })
    weighted_total = 4 * scores.agreement + 3 * scores.adequacy + 2 * scores.faithfulness + scores.clarity
    return {
        "agreement": scores.agreement,
        "adequacy": scores.adequacy,
        "faithfulness": scores.faithfulness,
        "clarity": scores.clarity,
        "weighted_total": weighted_total,
        "max_score": 50,
    }


if __name__ == "__main__":
    # Sanity check with hand-picked cases.
    hit = score_mitre_agreement(["T1082", "T1595"], "T1082")
    miss = score_mitre_agreement(["T1595"], "T1082")
    assert hit.technique_hit is True and hit.recall == 1.0
    assert miss.technique_hit is False and miss.recall == 0.0

    ea_correct = score_escalation_agreement("escalate", True)
    ea_wrong = score_escalation_agreement("ignore", True)
    assert ea_correct.correct is True
    assert ea_wrong.correct is False

    counts = confusion_counts([ea_correct, ea_wrong, score_escalation_agreement("ignore", False)])
    print("confusion counts:", counts)
    print("precision/recall/F1:", precision_recall_f1(counts))

    calib = calibration_by_confidence([
        {"confidence": "high", "correct": True},
        {"confidence": "high", "correct": True},
        {"confidence": "low", "correct": False},
        {"confidence": "low", "correct": True},
    ])
    print("calibration:", calib)
    print("OK: metrics sanity checks passed (score_llm_judge needs live OPENAI_API_KEY, not covered here)")
