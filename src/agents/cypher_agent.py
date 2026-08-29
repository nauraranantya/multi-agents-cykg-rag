# src/agents/cypher_agent.py
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from src.config.settings import read_only_graph
from src.retrieval.temporal import weight_and_sort_records

# --- Cypher Generation Prompt Template ---
# Schema + examples below match src/ingestion/graph_loader.py exactly (the
# actual graph-construction code in this repo, not the earlier no-code
# hosted tool the original AgCyRAG paper assumed -- see
# ../../defengraph_vs_agcyrag_analysis.md Sec.3.2). Earlier examples in this
# prompt referenced FAILED_LOGIN/SESSION_OPENED_ON/AUTHENTICATION_FAILURE_ON
# relationship types that don't exist anywhere in graph_loader.py -- a stale
# holdover from a different demo schema. Every relationship type below is
# one graph_loader.py actually writes.
cypher_generation_template = """
You are an expert Neo4j Cypher translator investigating security alerts, converting an
investigative question into Cypher based on the Neo4j Schema provided.

Before writing the query, first work out what data the question actually needs: (a) which entity
it's anchored on (a specific Host/IP/User/Alert), and (b) whether it needs a KNOWN, SPECIFIC filter
value (a literal ID, technique ID, or timestamp) or whether it needs the alert's own descriptive
content (title, full_log, rule_level) returned so that content itself can answer the question.
Retrieve BROADLY around the anchor entity and return descriptive fields for the latter case --
never invent a speculative WHERE-clause filter using a guessed keyword (e.g. filtering for the
word "scanning" or "probing" in some property) just because it sounds like it should match the
hypothesis. A query that returns more rows than strictly needed (for downstream review to read and
interpret) is far better than one that returns nothing because a guessed filter value doesn't
actually match what's stored -- Suricata/Wazuh alert titles and full_log text already describe
what was detected in human-readable form; that description IS the evidence, not a keyword filter
invented to pre-judge it.

Following the instructions below:
        1. Generate Cypher query compatible ONLY for Neo4j Version 5.
        2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use an alias when using the WITH keyword.
        3. Use only Node labels and Relationship types mentioned in the schema.
        4. Do not use relationships that are not mentioned in the given schema.
        5. For property searches, use case-insensitive matching. E.g., to search for a User, use `toLower(u.id) CONTAINS 'search_term'`.
        6. Assign a meaningful alias to every node and relationship in the MATCH clause (e.g., `MATCH (h:Host)-[r:HAS_ALERT]->(a:Alert)`).
        7. In the RETURN clause, include only the components (nodes, relationships, or properties) needed to answer the question.
        8. To count distinct items from an `OPTIONAL MATCH`, collect them first and then use `size()` on the list to avoid null value warnings (e.g., `WITH main, collect(DISTINCT opt) AS items RETURN size(items) AS itemCount`).
        9. To create unique pairs of nodes for comparison, use `WHERE elementId(node1) < elementId(node2)`.
        10. **CRITICAL RULE**: When returning the `type()` of a relationship, you MUST give the relationship a variable in the `MATCH` clause. E.g., `MATCH (h:Host)-[r:HAS_ALERT]->(a:Alert) RETURN type(r)`. Do NOT use `type()` on a relationship without a variable.
        11. **CRITICAL RULE**: A `WITH`/`RETURN` clause using `DISTINCT` or an aggregate function (`count()`, `collect()`, etc.) drops every variable not explicitly named in that clause -- a later `ORDER BY` can ONLY reference variables that clause actually returned, never an earlier MATCH variable that fell out of scope. E.g. `MATCH (h:Host)-[:HAS_ALERT]->(a:Alert) RETURN DISTINCT h.id AS hostId ORDER BY a.timestamp` is INVALID (`a` is gone after the DISTINCT); either return `a.timestamp` as an alias and order by that alias, or drop DISTINCT/the aggregate if you need to keep ordering by an unreturned variable.
        12. **CRITICAL RULE**: `MitreTechnique.tactic` is populated ONLY from the original alert's own native tag (most alerts don't have one), and when present it is always one of the ~14 standard MITRE ATT&CK tactic names verbatim (e.g. "Discovery", "Lateral Movement", "Initial Access", "Command and Control") -- it is NEVER a hypothesis's own free-text wording. Words like "scanning", "probing", or "reconnaissance" will NEVER appear in `m.tactic` even for genuinely related alerts, so `WHERE toLower(m.tactic) CONTAINS "scanning"` is guaranteed to return nothing regardless of whether relevant data exists. If the question names a SPECIFIC known technique, filter by `m.id` (e.g. `m.id = "T1046"` or `m.id STARTS WITH "T1046"`) -- that is the reliable, structured identifier. If the question is more open-ended (e.g. "does this look like scanning activity"), do NOT filter MitreTechnique by any guessed tactic text at all -- retrieve the alert's own `title`/`full_log` instead (per the guidance above) and let that content answer it.

Schema:
{schema}

Domain notes (this graph represents security alerts, not general logs):
- (Host)-[:HAS_ALERT]->(Alert): which host an alert fired on.
- (IP)-[:ATTACK_TO]->(Alert): the source/attacking IP for an alert, if any.
- (Alert)-[:CONNECTS_TO]->(IP): the destination IP for an alert, if any.
- (Alert)-[:TARGETS_USER]->(User): the targeted user account, if any.
- (Alert)-[:TRIGGERS]->(MitreTechnique {{id, tactic}}): MITRE ATT&CK technique(s) the alert was
  natively tagged with (not every alert has one -- absence is itself informative).
- Alert nodes carry: id, title, rule_id, rule_level (Wazuh severity, 0-15), timestamp (ISO-8601),
  full_log, scenario_id.
- Investigative questions about a security alert are usually one of: scope/correlation (what else
  is connected to this host/IP/user), temporal (what happened before/after, in what order),
  historical recurrence (has this host/IP/user triggered alerts before), or technique history
  (what MITRE techniques has this host/IP triggered). Prefer queries that traverse from the
  entity in question (Host/IP/User) outward, ordered by `a.timestamp`, over single-alert lookups.

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than for you to construct a Cypher statement.
Do not run any queries that would add to or delete from the database.

Examples:

1.  Question: What other alerts has this host triggered, in order?
    Query:
    MATCH (h:Host {{id: "webserver01"}})-[:HAS_ALERT]->(a:Alert)
    RETURN a.id AS alertId, a.title AS title, a.rule_level AS severity, a.timestamp AS timestamp
    ORDER BY a.timestamp

2.  Question: Has this source IP attacked any other hosts, and how many?
    Query:
    MATCH (ip:IP {{id: "203.0.113.7"}})-[:ATTACK_TO]->(a:Alert)<-[:HAS_ALERT]-(h:Host)
    RETURN count(DISTINCT h) AS hostsAffected, collect(DISTINCT h.id) AS hostIds

3.  Question: What MITRE ATT&CK techniques has this host triggered previously?
    Query:
    MATCH (h:Host {{id: "webserver01"}})-[:HAS_ALERT]->(a:Alert)-[:TRIGGERS]->(m:MitreTechnique)
    RETURN DISTINCT m.id AS techniqueId, m.tactic AS tactic, a.timestamp AS whenTriggered
    ORDER BY a.timestamp

4.  Question: Has this user been targeted by an alert before this timestamp?
    Query:
    MATCH (u:User {{id: "daryl"}})<-[:TARGETS_USER]-(a:Alert)
    WHERE a.timestamp < "2026-08-04T01:02:03Z"
    RETURN a.id AS alertId, a.title AS title, a.timestamp AS timestamp
    ORDER BY a.timestamp DESC
    LIMIT 5

5.  Question: Give me all activity connected to this alert.
    Query:
    MATCH (a:Alert {{id: "alert-123"}})
    OPTIONAL MATCH (h:Host)-[:HAS_ALERT]->(a)
    OPTIONAL MATCH (srcIp:IP)-[:ATTACK_TO]->(a)
    OPTIONAL MATCH (a)-[:CONNECTS_TO]->(dstIp:IP)
    OPTIONAL MATCH (a)-[:TARGETS_USER]->(u:User)
    OPTIONAL MATCH (a)-[:TRIGGERS]->(m:MitreTechnique)
    RETURN a.title AS title, a.rule_level AS severity, h.id AS host, srcIp.id AS sourceIp,
           dstIp.id AS destIp, u.id AS targetUser, collect(DISTINCT m.id) AS techniques

6.  Question: Is there evidence of outbound scanning or probing activity from this IP?
    Query (WRONG way -- do not do this: `WHERE toLower(m.tactic) CONTAINS "scanning"` guesses at
    free-text that is never actually stored, guaranteeing an empty result either way):
    MATCH (ip:IP {{id: "10.143.2.4"}})-[:ATTACK_TO]->(a:Alert)
    WHERE toLower(a.title) CONTAINS "scan" OR toLower(a.full_log) CONTAINS "scan"
    RETURN a.id AS alertId, a.title AS title, a.rule_level AS severity, a.timestamp AS timestamp,
           a.full_log AS log
    ORDER BY a.timestamp
    (Right approach: retrieve every alert this IP triggered -- with its title/full_log, which
    already describes what was actually detected -- rather than pre-filtering on a guessed keyword
    that might not appear verbatim anywhere. If the volume is large, a light keyword filter on
    `a.title`/`a.full_log` themselves, as above, is fine since those fields do contain real
    descriptive text -- the mistake is filtering `m.tactic` on words that field never contains.)


The question is:
{question}
"""

cyper_generation_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema","question"]
)

# --- Cypher QA Prompt Template ---
qa_template = """
You are a security analyst assistant that takes the results from a Neo4j security-alert graph query and forms a human-readable response. The query results section contains the results of a Cypher query that was generated based on an investigative question about a security alert. The provided information is authoritative; you must never question it or use your internal knowledge to alter it. Make the answer sound like a response to the question.
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
            # A dedicated read-only connection (src/config/settings.py's
            # ReadOnlyNeo4jGraph), not the shared `graph` -- this is the
            # one place an LLM's own generated Cypher gets executed
            # against the live graph; see SECURITY_ASSESSMENT.md.
            graph=read_only_graph.resolve(),
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