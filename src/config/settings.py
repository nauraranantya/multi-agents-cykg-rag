# src/config/settings.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

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


def _build_graph() -> Neo4jGraph:
    uri = os.environ.get("NEO4J_AURA")
    username = os.environ.get("NEO4J_AURA_USERNAME")
    password = os.environ.get("NEO4J_AURA_PASSWORD")
    database = os.environ.get("NEO4J_AURA_DATABASE") or "neo4j"
    if not (uri and username and password):
        raise RuntimeError("NEO4J_AURA / NEO4J_AURA_USERNAME / NEO4J_AURA_PASSWORD are not set")
    return Neo4jGraph(url=uri, username=username, password=password, database=database)


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