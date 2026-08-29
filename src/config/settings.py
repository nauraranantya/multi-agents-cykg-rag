# src/config/settings.py
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Set before mcp_use is ever imported anywhere in this codebase (only
# src/agents/mcp_rdf_agent.py imports it, but settings.py is imported by
# every agent module first, so this always runs earlier). mcp_use's
# MCPAgent constructs a Posthog+Scarf telemetry client on every
# instantiation unless this is already "false" by then --
# mcp_rdf_agent.py's own get_mcp_client() sets it too, but only right
# before building the MCPClient in that one function; setting it here as
# well means it's never a race against import order elsewhere. Only
# `setdefault` -- an explicit MCP_USE_ANONYMIZED_TELEMETRY=true in .env is
# still honored.
os.environ.setdefault("MCP_USE_ANONYMIZED_TELEMETRY", "false")

# --- Optional LangChain/LangSmith tracing env vars ---
# Only forward these if actually set -- previously this did
# `os.environ["X"] = os.environ.get("X")`, which raised TypeError (str
# expected, not NoneType) the instant ANY of these were unset, crashing
# every import of this module before Neo4j was ever reached.
for _var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = _val

DEFAULT_MAX_ITERATIONS = 3
VECTOR_INDEX_NAME = "vector"
KEYWORD_INDEX_NAME = "keyword"
ENTITY_FULLTEXT_INDEX_NAME = "entities"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output size


class _LazyProxy:
    """Defers construction of a networked object (Neo4j driver, vector index)
    until it's actually used. Neo4jGraph/Neo4jVector dial out and raise
    inside their own constructors, so without this, importing ANY module
    that does `from src.config.settings import graph` -- including ones that
    never touch Neo4j, like the guardrail or MCP RDF agents -- crashed the
    whole app on a missing credential or a paused Aura free-tier instance.
    Now the failure happens on first real use, with a clear message."""

    def __init__(self, factory, label):
        self._factory = factory
        self._label = label
        self._instance = None

    def _get(self):
        if self._instance is None:
            try:
                self._instance = self._factory()
            except Exception as exc:
                raise RuntimeError(
                    f"{self._label} is not available: {exc}\n"
                    f"Check NEO4J_AURA / NEO4J_AURA_USERNAME / NEO4J_AURA_PASSWORD "
                    f"(and NEO4J_AURA_DATABASE, if your instance's default database "
                    f"isn't named 'neo4j') in multi-agents-cykg-rag/.env."
                ) from exc
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def resolve(self):
        """Return the real underlying object. Needed at call-sites that
        require actual isinstance conformance (e.g. GraphCypherQAChain's
        pydantic-validated `graph: GraphStore` field) rather than the
        duck-typed attribute proxying __getattr__ provides -- isinstance()
        checks don't go through __getattr__."""
        return self._get()


def _build_llm() -> ChatOpenAI:
    # A missing/empty OPENAI_API_KEY makes ChatOpenAI's constructor itself
    # raise (it eagerly builds an openai.OpenAI client). Passing an obvious
    # placeholder when unset lets construction succeed everywhere so agent
    # modules can still be imported; the real failure then naturally surfaces
    # as an auth error on first actual `.invoke()`, not at import time.
    api_key = os.environ.get("OPENAI_API_KEY") or "sk-not-configured"
    return ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=api_key)


def _neo4j_conn_kwargs() -> dict:
    uri = os.environ.get("NEO4J_AURA")
    username = os.environ.get("NEO4J_AURA_USERNAME")
    password = os.environ.get("NEO4J_AURA_PASSWORD")
    database = os.environ.get("NEO4J_AURA_DATABASE") or "neo4j"
    if not (uri and username and password):
        raise RuntimeError("NEO4J_AURA / NEO4J_AURA_USERNAME / NEO4J_AURA_PASSWORD are not set")
    return {"url": uri, "username": username, "password": password, "database": database}


def _build_graph() -> Neo4jGraph:
    # driver_config's connection_timeout bounds the *first* connection
    # attempt specifically -- a paused free-tier Aura instance (auto-pauses
    # after inactivity, per _LazyProxy's own docstring above) otherwise
    # leaves the driver's default retry/backoff to decide how long to hang
    # before giving up, which can look indistinguishable from a genuine
    # deadlock. This makes that failure mode fail fast with a clear
    # RuntimeError (via _LazyProxy._get's except clause) instead.
    return Neo4jGraph(**_neo4j_conn_kwargs(), driver_config={"connection_timeout": 15})


# Cypher/DDL clauses that mutate data or schema. Word-boundary matched,
# case-insensitive; covers the standard write clauses plus the write-side
# APOC procedure families (apoc.create.*, apoc.merge.*, apoc.periodic.*,
# etc. -- APOC ships plenty of read-only procedures too, so this only
# blocks `CALL apoc.<family>.` names that are themselves write-shaped).
_WRITE_CYPHER_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|"
    r"CALL\s+apoc\.[a-zA-Z]+\.(create|merge|delete|remove|refactor|periodic|write))\b",
    re.IGNORECASE,
)


class ReadOnlyNeo4jGraph(Neo4jGraph):
    """Defense-in-depth for the one place in this codebase an LLM's own
    generated Cypher gets executed against the live graph:
    src/agents/cypher_agent.py's GraphCypherQAChain (built with
    allow_dangerous_requests=True, per langchain_neo4j's own required
    opt-in) calls self.graph.query(generated_cypher) with no execution-time
    restriction of its own -- see langchain_neo4j/chains/graph_qa/cypher.py
    _call(). The prompt there already says "do not run queries that would
    add to or delete from the database", but that's a prompt-level ask,
    not a technical control -- trivially bypassable by a bad generation or
    a prompt-injected alert field (full_log/rule_description) flowing into
    `question`. This is the real capability-scoping boundary (see
    SECURITY_ASSESSMENT.md): reject any query containing a write/schema
    clause before it ever reaches Neo4j. A *separate* connection object
    from the shared `graph` below, which legitimately needs write access
    for ingestion (src/ingestion/graph_loader.py) -- only
    src/agents/cypher_agent.py should ever use this one."""

    def query(self, query: str, params: dict = {}, session_params: dict = {}):
        if _WRITE_CYPHER_RE.search(query):
            raise ValueError(
                f"Refusing to execute a write/schema-mutating Cypher query "
                f"against the read-only connection: {query!r}"
            )
        return super().query(query, params=params, session_params=session_params)


def _build_read_only_graph() -> ReadOnlyNeo4jGraph:
    return ReadOnlyNeo4jGraph(**_neo4j_conn_kwargs(), driver_config={"connection_timeout": 15})


_embeddings_cache: dict = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Shared, lazily-constructed embeddings model -- used by both the
    vector index below and src/ingestion/graph_loader.py, so the model is
    only loaded once per process."""
    if "value" not in _embeddings_cache:
        _embeddings_cache["value"] = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_cache["value"]


def _build_vector_index() -> Neo4jVector:
    uri = os.environ.get("NEO4J_AURA")
    username = os.environ.get("NEO4J_AURA_USERNAME")
    password = os.environ.get("NEO4J_AURA_PASSWORD")
    database = os.environ.get("NEO4J_AURA_DATABASE") or "neo4j"
    if not (uri and username and password):
        raise RuntimeError("NEO4J_AURA / NEO4J_AURA_USERNAME / NEO4J_AURA_PASSWORD are not set")
    return Neo4jVector.from_existing_index(
        embedding=get_embeddings(),
        url=uri,
        username=username,
        password=password,
        database=database,
        index_name=VECTOR_INDEX_NAME,
        keyword_index_name=KEYWORD_INDEX_NAME,
        search_type="hybrid",
    )


# --- LLM init (always a real ChatOpenAI; see _build_llm for why this is safe) ---
llm = _build_llm()

# --- Neo4j graph + vector index (lazy: connects on first real use) ---
graph = _LazyProxy(_build_graph, "Neo4j graph connection")
vector_index = _LazyProxy(_build_vector_index, "Neo4j vector index")
# Separate connection, separate singleton: see ReadOnlyNeo4jGraph's
# docstring above -- only src/agents/cypher_agent.py should use this one.
read_only_graph = _LazyProxy(_build_read_only_graph, "Neo4j graph connection (read-only)")

_schema_cache: dict = {}


def get_schema_escaped() -> str:
    """Neo4j schema, brace-escaped for use inside an f-string/prompt template.
    Cached after first (lazy) fetch. Replaces the old module-level
    NEO4J_SCHEMA_ESCAPED_FOR_PROMPT constant, which forced a live schema
    query at import time."""
    if "value" not in _schema_cache:
        raw = graph.schema
        _schema_cache["value"] = raw.replace("{", "{{").replace("}", "}}")
    return _schema_cache["value"]