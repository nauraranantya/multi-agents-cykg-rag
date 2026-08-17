# src/agents/vector_agent.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from pydantic import BaseModel, Field
from typing import List, Optional
from src.config.settings import llm, graph, vector_index
from src.retrieval.temporal import weight_and_sort_records, weight_and_sort_documents

# --- Entity Extraction ---
class LogEntities(BaseModel):
    """Identifies information about resources in the log."""

    entity_values: List[str] = Field(
        ...,
        description="All entities such as User, Server, Service, Host, "
        "System, Software, Device, Process, Machine, Session, or Document file names that appear in the text.",
    )

entity_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at extracting entities from text related to system logs. "
            "Extract the name or ID of entities such as User, Server, Service, "
            "Host, System, Software, Device, Process, Machine, Session, and the filename of the Document.",
        ),
        (
            "human",
            "Use the given format to extract information from"
            "the following input: {question}",
        ),
    ]
)

entity_chain = entity_prompt | llm.with_structured_output(LogEntities)

# --- Helper Functions ---
def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(question: str, query_timestamp: Optional[str] = None) -> str:
    """
    Collects the neighborhood of resources mentioned
    in the question
    """
    result = ""

    entities = entity_chain.invoke({"question": question})
    print(f"\n--- Extracted Entities: {entities.entity_values} ---")

    for entity_value in entities.entity_values:
        query = generate_full_text_query(entity_value)
        if not query:
            continue

        # Raw fields, not a pre-formatted string, so results can be
        # temporally re-ranked (weight_and_sort_records) before display.
        rows = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entities', $query, {limit: 10})
            YIELD node AS entity

            MATCH (chunk:Chunk)-[:HAS_ENTITY]->(entity)

            OPTIONAL MATCH (chunk)-[:PART_OF]->(doc:Document)

            WITH entity, chunk, doc,
                 CASE WHEN 'Document' IN labels(entity)
                      THEN entity.fileName
                      ELSE entity.id
                 END AS entity_name

            RETURN entity_name AS entity_name,
                   coalesce(doc.fileName, 'N/A') AS doc_name,
                   left(chunk.text, 250) AS chunk_text,
                   chunk.timestamp AS event_timestamp
            LIMIT 10
            """,
            {"query": query},
        )
        if query_timestamp:
            rows = weight_and_sort_records(rows, query_timestamp, preferred_keys=("event_timestamp",))
        if rows:
            result += "\n".join(
                f"Entity '{row['entity_name']}' found in document '{row['doc_name']}'. "
                f"The context of the text is: '{row['chunk_text']}...'"
                for row in rows
            )
    return result

# --- Main Search Function ---
def query_vector_search(question: str, query_timestamp: Optional[str] = None):
    """
    Query the graph and vector index using a vector approach for vector similarity search.
    This is for questions that require finding similar concepts or descriptions.

    `query_timestamp`, when given (the triggering alert's timestamp for
    auto-triggered queries, "now" for manual ones), re-ranks both the
    structured and unstructured results by recency (src/retrieval/temporal.py,
    ported from DefenGraph eq. 5) so recent events outrank stale ones with
    otherwise-similar relevance.
    """
    print(f"--- Executing Vector Search for: {question} ---")
    structured_data = structured_retriever(question, query_timestamp)
    unstructured_docs = vector_index.similarity_search(question)
    if query_timestamp:
        unstructured_docs = weight_and_sort_documents(unstructured_docs, query_timestamp)
    unstructured_data = [el.page_content for el in unstructured_docs]
    final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Resource ". join(unstructured_data)}
    """
    return final_data