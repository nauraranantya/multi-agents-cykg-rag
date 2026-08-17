# src/agents/cypher_agent.py
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from src.config.settings import graph
from src.retrieval.temporal import weight_and_sort_records

# --- Cypher Generation Prompt Template ---
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
        1. Generate Cypher query compatible ONLY for Neo4j Version 5.
        2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use an alias when using the WITH keyword.
        3. Use only Node labels and Relationship types mentioned in the schema.
        4. Do not use relationships that are not mentioned in the given schema.
        5. For property searches, use case-insensitive matching. E.g., to search for a User, use `toLower(u.id) CONTAINS 'search_term'`.
        6. Assign a meaningful alias to every node and relationship in the MATCH clause (e.g., `MATCH (u:User)-[r:FAILED_LOGIN]->(s:System)`).
        7. In the RETURN clause, include only the components (nodes, relationships, or properties) needed to answer the question.
        8. To count distinct items from an `OPTIONAL MATCH`, collect them first and then use `size()` on the list to avoid null value warnings (e.g., `WITH main, collect(DISTINCT opt) AS items RETURN size(items) AS itemCount`).
        9. To create unique pairs of nodes for comparison, use `WHERE elementId(node1) < elementId(node2)`.
        10. **CRITICAL RULE**: When returning the `type()` of a relationship, you MUST give the relationship a variable in the `MATCH` clause. E.g., `MATCH (u:User)-[r:HAS_SESSION]->(s:Server) RETURN type(r)`. Do NOT use `type()` on a relationship without a variable.

Schema:
{schema}

Note: 
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than for you to construct a Cypher statement.
Do not run any queries that would add to or delete from the database.

Examples:

1.  Question: Which users have the most authentication failures?
    Query:
    MATCH (u:User)-[:AUTHENTICATION_FAILURE_ON]->()
    RETURN u.id AS userId, count(*) AS failureCount
    ORDER BY failureCount DESC
    LIMIT 10

2.  Question: List devices where users opened or closed a session.
    Query:
    MATCH (u:User)-[r:SESSION_OPENED_ON|SESSION_CLOSED_ON]->(device)
    RETURN u.id AS userId, type(r) AS action, labels(device) AS deviceType, device.id AS deviceId
    LIMIT 20

3.  Question: Tell the full path of the session: from the device where it was opened to where it was closed by root user
    Query:
    MATCH (u:User {{id: "root"}})-[open:SESSION_OPENED_ON]->(startDevice),(u)-[close:SESSION_CLOSED_ON]->(endDevice)
    RETURN
        u.id            AS userId,
        type(open)      AS openedOnRel,
        labels(startDevice) AS startDeviceType,
        startDevice.id  AS startDeviceId,
        type(close)     AS closedOnRel,
        labels(endDevice)   AS endDeviceType,
        endDevice.id    AS endDeviceId
        
4.  Question: Give me information about daryl's activity?
    Query:
    MATCH (u:User)-[r]->(n)
    WHERE toLower(u.id) = 'daryl'
    RETURN u.id AS user, type(r) as relationship, n.id as entity


The question is:
{question}
"""

cyper_generation_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema","question"]
)

# --- Cypher QA Prompt Template ---
qa_template = """
You are an assistant that takes the results from a Neo4j Cypher query and forms a human-readable response. The query results section contains the results of a Cypher query that was generated based on a user's natural language question. The provided information is authoritative; you must never question it or use your internal knowledge to alter it. Make the answer sound like a response to the question.
Final answer should be easily readable and structured.
Query Results:
{context}

Question: {question}
If the provided information is empty, respond by stating that you don't know the answer. Empty information is indicated by: []
If the information is not empty, you must provide an answer using the results. If the question involves a time duration, assume the query results are in units of days unless specified otherwise.
Never state that you lack sufficient information if data is present in the query results. Always utilize the data provided.
Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"]
)

# --- Cypher QA Chain and Query Function ---
# Built lazily (not at import time): GraphCypherQAChain.from_llm() eagerly
# calls graph.get_structured_schema, which -- now that `graph` is a lazy
# Neo4j proxy (see src/config/settings.py) -- would otherwise force a live
# Neo4j connection the instant this module is imported, even by code that
# never calls query_cypher() at all.
_cypher_qa_chain = None


def _get_cypher_qa_chain() -> GraphCypherQAChain:
    global _cypher_qa_chain
    if _cypher_qa_chain is None:
        _cypher_qa_chain = GraphCypherQAChain.from_llm(
            top_k=10,
            graph=graph.resolve(),
            verbose=True,
            validate_cypher=True,
            return_intermediate_steps=True,
            cypher_prompt=cyper_generation_prompt,
            qa_prompt=qa_generation_prompt,
            qa_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            cypher_llm=ChatOpenAI(model="gpt-4o", temperature=0),
            allow_dangerous_requests=True,
            use_function_response=True
        )
    return _cypher_qa_chain


def query_cypher(question: str, query_timestamp: Optional[str] = None) -> dict:
    """
    Generate and run a Cypher query against the graph database.
    Use this for complex questions requiring structured data, aggregations, or specific graph traversals
    Returns the query and the result context.

    `query_timestamp`, when given, re-ranks the result rows by recency
    (src/retrieval/temporal.py) before they're returned -- the generated
    Cypher's RETURN clause is LLM-controlled and not fixed-schema, so rows
    are heuristically scanned for an ISO-8601 timestamp value rather than
    assuming a specific column.
    """
    print(f"--- Executing Cypher Search for: {question} ---")
    response = _get_cypher_qa_chain().invoke({"query": question})
    context = response["intermediate_steps"][1]["context"]
    if query_timestamp and isinstance(context, list):
        context = weight_and_sort_records(context, query_timestamp)
    return {
        "query": response["intermediate_steps"][0]["query"],
        "context": context
    }