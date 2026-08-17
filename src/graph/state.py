# src/graph/state.py
from typing import List, Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    question: str
    original_question: str
    is_relevant: bool
    is_log_question: bool
    is_cskg_required: bool

    log_vector_context: Optional[str]
    log_cypher_context: Optional[List[dict]]

    latest_vector_context: Optional[List[dict]]
    latest_cypher_context: Optional[List[dict]]

    generated_question_for_rdf: Optional[str]
    mcp_rdf_context: Optional[str]

    answer: Optional[str]
    cypher_query: Optional[str]
    error: Optional[str]
    messages: Annotated[list, add_messages]

    # reflection state
    cypher_iteration_count: int
    vector_iteration_count: int
    vector_answer_sufficient: bool
    cypher_answer_sufficient: bool

    max_iterations: int

    # structured synthesizer output + grounding check
    synthesized_report: Optional[dict]
    grounding_result: Optional[dict]
    grounding_retry_count: int

    # temporal weighting: the timestamp retrieval should be weighted against
    # (the triggering alert's timestamp for auto-triggered queries, "now"
    # for manual analyst questions)
    query_timestamp: Optional[str]