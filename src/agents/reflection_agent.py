# ### src/agents/reflection_agents.py ###
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import llm

class RephrasedQuestion(BaseModel):
    rephrased_question: str = Field(description="A rephrased, more specific version of the original question to improve answer generation.")

# --- Vector Reflection ---
vector_reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a query correction expert. A vector search returned insufficient or irrelevant context.
        Your task is to rephrase the user's question to be more specific and likely to succeed with a vector/keyword search.
        Analyze the original question and the insufficient context. For example, if the question was too broad, suggest adding specific keywords. If it used ambiguous terms, make them clearer.
        Do not just repeat the question. Provide a meaningful improvement.
        """
    ),
    (
        "human",
        "Original Question: {original_question}\n\nInsufficient Context from Vector Search:\n{log_vector_context}\n\nRephrase the question to improve the chances of getting a better result."
    ),
])
vector_reflection_chain = vector_reflection_prompt | llm.with_structured_output(RephrasedQuestion)
# --- Cypher Reflection ---
cypher_reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a query correction expert. A Cypher query returned no results.
        Your task is to rephrase the user's question to be more specific and likely to succeed with the given Neo4j graph schema.
        Analyze the failed query and the schema. For example, if the question was too broad, make it more specific. If it used terms not in the schema, suggest alternatives.
        Do not just repeat the question. Provide a meaningful improvement.

        Schema:
        {schema}
        """
    ),
    (
        "human", 
        "Original Question: {original_question}\n\nFailed Cypher Query:\n{cypher_query}\n\nRephrase the question to improve the chances of getting a result."
    ),
])
reflection_chain = cypher_reflection_prompt | llm.with_structured_output(RephrasedQuestion)