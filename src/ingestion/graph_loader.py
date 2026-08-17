# src/ingestion/graph_loader.py
"""
Raw alert -> Neo4j graph construction. This is the concrete fix for the
reproducibility gap flagged in ../../defengraph_vs_agcyrag_analysis.md
Sec.3.2: no graph-construction code existed anywhere in the AgCyRAG repo --
the Neo4j LPG was assumed to be built by a separate, external, no-code
hosted tool ("Neo4j LLM Graph Builder").

Produces two layers in the same graph:
  1. A structured entity graph (Host/IP/User/Alert/MitreTechnique nodes +
     relationships, each carrying a `timestamp`) that
     src/agents/cypher_agent.py's LLM-generated Cypher queries against.
  2. A Chunk/Document layer (Chunk --PART_OF--> Document,
     Chunk --HAS_ENTITY--> entity) matching the schema
     src/agents/vector_agent.py already assumes exists (a fulltext
     "entities" index, hybrid vector+keyword search over Chunk text) -- one
     Chunk per alert, embedding a natural-language rendering of it.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from src.config.settings import (
    graph,
    get_embeddings,
    VECTOR_INDEX_NAME,
    KEYWORD_INDEX_NAME,
    ENTITY_FULLTEXT_INDEX_NAME,
    EMBEDDING_DIMENSION,
)
from src.ingestion.schema import Alert

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def ensure_indices() -> None:
    """Idempotent -- safe to call on every run/every alert batch."""
    graph.query(
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSION},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
    )
    graph.query(
        f"CREATE FULLTEXT INDEX {KEYWORD_INDEX_NAME} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
    )
    graph.query(
        f"""
        CREATE FULLTEXT INDEX {ENTITY_FULLTEXT_INDEX_NAME} IF NOT EXISTS
        FOR (n:Host|IP|User|MitreTechnique|Document) ON EACH [n.id, n.fileName]
        """
    )
    for label, prop in [
        ("Alert", "id"), ("Host", "id"), ("IP", "id"), ("User", "id"),
        ("MitreTechnique", "id"), ("Chunk", "id"), ("Document", "fileName"),
    ]:
        graph.query(
            f"CREATE CONSTRAINT {label.lower()}_{prop} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )


def _alert_narrative(alert: Alert) -> str:
    bits = [f"Alert {alert.id} on host {alert.agent_name} at {alert.timestamp}: {alert.rule_description}"]
    if alert.data_srcip:
        bits.append(f"source IP {alert.data_srcip}")
    if alert.data_dstip:
        bits.append(f"destination IP {alert.data_dstip}")
    if alert.data_dstuser:
        bits.append(f"user {alert.data_dstuser}")
    if alert.rule_mitre_id:
        bits.append("MITRE ATT&CK techniques: " + ", ".join(alert.rule_mitre_id))
    bits.append(f"Raw log: {alert.full_log}")
    return ". ".join(bits)


_INGEST_ALERT_CYPHER = """
MERGE (h:Host {id: $agent_name})
MERGE (a:Alert {id: $id})
  SET a.title = $title, a.rule_id = $rule_id, a.rule_level = $rule_level,
      a.timestamp = $timestamp, a.full_log = $full_log, a.scenario_id = $scenario_id
MERGE (h)-[:HAS_ALERT {timestamp: $timestamp}]->(a)

MERGE (d:Document {fileName: $id})
  SET d.timestamp = $timestamp
MERGE (c:Chunk {id: $id + '-chunk'})
  SET c.text = $chunk_text, c.embedding = $chunk_embedding, c.timestamp = $timestamp
MERGE (c)-[:PART_OF]->(d)
MERGE (c)-[:HAS_ENTITY]->(h)

WITH a, c
FOREACH (srcip IN CASE WHEN $srcip IS NULL THEN [] ELSE [$srcip] END |
    MERGE (ip:IP {id: srcip})
    MERGE (ip)-[:ATTACK_TO {timestamp: $timestamp}]->(a)
    MERGE (c)-[:HAS_ENTITY]->(ip)
)
FOREACH (dstip IN CASE WHEN $dstip IS NULL THEN [] ELSE [$dstip] END |
    MERGE (ip2:IP {id: dstip})
    MERGE (a)-[:CONNECTS_TO {timestamp: $timestamp}]->(ip2)
    MERGE (c)-[:HAS_ENTITY]->(ip2)
)
FOREACH (dstuser IN CASE WHEN $dstuser IS NULL THEN [] ELSE [$dstuser] END |
    MERGE (u:User {id: dstuser})
    MERGE (a)-[:TARGETS_USER {timestamp: $timestamp}]->(u)
    MERGE (c)-[:HAS_ENTITY]->(u)
)
"""

_INGEST_TECHNIQUE_CYPHER = """
MERGE (m:MitreTechnique {id: $tid})
  SET m.tactic = coalesce($tactic, m.tactic)
WITH m
MATCH (a:Alert {id: $alert_id}), (c:Chunk {id: $chunk_id})
MERGE (a)-[:TRIGGERS {timestamp: $timestamp}]->(m)
MERGE (c)-[:HAS_ENTITY]->(m)
"""


def ingest_alert(alert: Alert) -> None:
    """Idempotent: safe to call more than once for the same alert id."""
    chunk_text = _alert_narrative(alert)
    chunk_embedding = get_embeddings().embed_query(chunk_text)

    graph.query(
        _INGEST_ALERT_CYPHER,
        {
            "id": alert.id,
            "agent_name": alert.agent_name,
            "title": alert.rule_description,
            "rule_id": alert.rule_id,
            "rule_level": alert.rule_level,
            "timestamp": alert.timestamp,
            "full_log": alert.full_log,
            "scenario_id": alert.scenario_id,
            "chunk_text": chunk_text,
            "chunk_embedding": chunk_embedding,
            "srcip": alert.data_srcip,
            "dstip": alert.data_dstip,
            "dstuser": alert.data_dstuser,
        },
    )

    for tid, tactic in itertools.zip_longest(alert.rule_mitre_id, alert.rule_mitre_tactic, fillvalue=None):
        if tid is None:
            continue
        graph.query(
            _INGEST_TECHNIQUE_CYPHER,
            {
                "tid": tid,
                "tactic": tactic,
                "alert_id": alert.id,
                "chunk_id": f"{alert.id}-chunk",
                "timestamp": alert.timestamp,
            },
        )


def load_all(path: Path | None = None) -> int:
    path = path or (DATA_DIR / "alerts.jsonl")
    ensure_indices()
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ingest_alert(Alert(**json.loads(line)))
            n += 1
    return n


if __name__ == "__main__":
    n = load_all()
    print(f"Ingested {n} alerts into Neo4j (nodes: Host/IP/User/Alert/MitreTechnique/Chunk/Document).")
