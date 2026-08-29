# src/agents/review_agent.py
"""
Reviews evidence retrieved for one investigative question against the
hypothesis it was generated to discriminate (see question_generation_agent.py)
-- not just whether *some* fact came back.

Previously this was a flat sufficient/insufficient check whose only purpose
was to gate an immediate rephrase-and-retry (reflection_agent.py). That
retry loop is dissolved: src/agents/question_generation_agent.py already
asks several distinct, well-targeted questions up front, so a single
question coming back empty is no longer treated as a failure to recover
from -- it's potentially itself the answer (e.g. "no prior alerts from
this IP" genuinely rules out a "known repeat offender" hypothesis). What
the system needs from review now is not "should we retry" but "what does
this evidence say about the hypothesis," so the synthesizer can weigh
confirmed/ruled-out/inconclusive verdicts across all questions at once.
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import llm


class ReviewOutput(BaseModel):
    sufficient: bool = Field(
        description="Does the context contain at least one concrete, factual data point relevant "
        "to the question -- including a clear negative (e.g. 'zero results found')? Only false if "
        "the context is empty/erroneous in a way that gives no information either way."
    )
    hypothesis_verdict: Literal["confirmed", "ruled_out", "inconclusive"] = Field(
        description="Does this evidence confirm the hypothesis, rule it out, or remain inconclusive "
        "either way? Judge based on what the result -- including an empty one -- actually implies "
        "for THIS specific hypothesis, not just whether data was returned. E.g. zero prior alerts "
        "for a host is 'ruled_out' evidence against a 'recurring known issue' hypothesis, not "
        "'inconclusive' just because the list is empty."
    )
    reasoning: str = Field(description="One or two sentences justifying both fields.")


review_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a SOC analyst evaluating evidence gathered for one investigative question, given "
        "the specific hypothesis that question was meant to discriminate. Decide (a) whether the "
        "retrieved context contains a usable factual data point -- a clear negative counts -- and "
        "(b) whether it confirms, rules out, or leaves inconclusive the stated hypothesis. Do not "
        "default to 'inconclusive' just because the result is short or empty: reason about what "
        "that specific result implies for this specific hypothesis."
    ),
    (
        "human",
        "Hypothesis: {hypothesis}\n\nQuestion: {question}\n\nContext:\n{context}\n\n"
        "Evaluate sufficiency and the hypothesis verdict.",
    ),
])
review_chain = review_prompt | llm.with_structured_output(ReviewOutput)
