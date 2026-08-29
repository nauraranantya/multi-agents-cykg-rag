# src/chains/synthesizer.py
"""
`cited_entities`/`mitre_techniques` are read by src/agents/grounding_check.py
to verify the report only references entities actually present in the
retrieved context.

Also now consumes `hypothesis_summary` (built from question_generation_agent's
QuestionSet + review_agent's per-question verdicts, rendered in
src/graph/workflow.py::_render_hypothesis_summary) -- a deterministic
rollup of which candidate scenarios were confirmed/ruled out/left
inconclusive. This absorbs log_analysis_agent.py's old summarization role
(that module is dissolved -- its other job, deciding whether a CSKG lookup
was needed, is now decided per-question by question_generation_agent's
`source` tag instead of after the fact).
"""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import llm


class SynthesizedReport(BaseModel):
    original_question: str = Field(description="The user's original question, restated.")
    cypher_context_summary: str = Field(description="Summary of the Cypher/graph query context, or 'Not applicable for this query.'")
    vector_context_summary: str = Field(description="Summary of the vector/full-text search context, or 'Not applicable for this query.'")
    generated_question_for_rdf: str = Field(description="The question generated for the cybersecurity knowledge base, or 'Not applicable for this query.'")
    cskg_context_summary: str = Field(description="Summary of the cybersecurity knowledge base (RDF/MITRE) context, or 'Not applicable for this query.'")
    critical_analysis: str = Field(description="How the information from all sources connects; whether a single source is sufficient on its own.")
    contextual_linkage: str = Field(description="The logical flow of the investigation, e.g. how a log finding led to a knowledge-base lookup.")
    final_answer: str = Field(description="The final, well-structured, human-readable answer for the analyst.")
    cited_entities: List[str] = Field(
        default_factory=list,
        description="Every specific entity identifier referenced in critical_analysis/final_answer "
        "(hostnames, IPs, usernames, ticket/document names) -- used to verify the answer is grounded "
        "in the retrieved context, not invented.",
    )
    mitre_techniques: List[str] = Field(
        default_factory=list,
        description="MITRE ATT&CK technique IDs (e.g. 'T1110') referenced in the analysis.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in final_answer given how much of the available evidence actually supports it. "
        "Note: this is capped downward by src/graph/workflow.py based on how many of the 3 context sources "
        "(Cypher/vector/CSKG) actually returned substantive data -- don't default to 'high' out of habit; a "
        "genuinely low-evidence answer should be rated 'low' or 'medium' here regardless of the cap."
    )
    recommended_priority: Literal["ignore", "monitor", "escalate"] = Field(
        description="Triage recommendation for a SOC analyst: 'escalate' if this warrants immediate "
        "human investigation/response, 'monitor' if it's worth tracking but not urgent, 'ignore' if it's "
        "benign/expected activity or there's no log/alert question to triage at all (e.g. a general "
        "cybersecurity-knowledge question)."
    )
    mitigation_suggestions: List[str] = Field(
        default_factory=list,
        description="Concrete, actionable response/mitigation steps a SOC analyst could take given "
        "final_answer -- e.g. specific accounts to disable, hosts to isolate, rules to tighten. Empty "
        "if recommended_priority is 'ignore' or there's nothing actionable to suggest (e.g. a general "
        "knowledge question)."
    )


synthesis_prompt = ChatPromptTemplate.from_template("""You are an expert cybersecurity analyst creating a final report.
Your task is to synthesize information gathered across several investigative questions -- each
generated to confirm or rule out a specific hypothesis about this alert -- into one report.

If a section's source has no data (e.g. no log data was queried), state "Not applicable for this query." for that field.

CRITICAL: only put entities (hostnames, IPs, usernames, MITRE technique IDs, etc.) into cited_entities/mitre_techniques
or reference them in critical_analysis/final_answer if they literally appear in the context below. Do not invent
or infer entities that are not present in the provided context.

Use the hypothesis summary below as your primary reasoning scaffold: weigh which hypotheses were
confirmed, ruled out, or left inconclusive, and let that drive critical_analysis and final_answer --
don't just restate raw context. A hypothesis being ruled out is as informative as one being confirmed.

Also set recommended_priority: 'escalate' for genuinely suspicious/malicious activity needing prompt human
response, 'monitor' for activity worth tracking but not urgent, 'ignore' for benign/expected activity or
questions that aren't about a specific alert at all (e.g. general cybersecurity knowledge questions).
Populate mitigation_suggestions with concrete response actions when recommended_priority is 'monitor' or
'escalate'; leave it empty otherwise.

If "Conversation so far" below is not N/A, this is a follow-up question inside an ongoing investigation --
final_answer should directly address the analyst's latest question in light of what was already found,
not restate the whole case from scratch.

**Case Context (static case fact sheet, if this belongs to a clustered investigation case):**
{case_context}

**Conversation So Far (prior turns in this investigation, if any):**
{conversation_context}

**Original Question:**
{original_question}

**Hypothesis Summary (generated questions + review verdicts):**
{hypothesis_summary}

**Cypher Log Information Context:**
{log_cypher_context}

**Vector Log Information Context:**
{log_vector_context}

**Generated Question for Cybersecurity Knowledge Base:**
{generated_question_for_rdf}

**Cybersecurity Knowledge Base Context (from RDF Agent):**
{mcp_rdf_context}
{retry_note}
""")

synthesis_chain = synthesis_prompt | llm.with_structured_output(SynthesizedReport)
