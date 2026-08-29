# src/agents/mcp_rdf_agent.py
"""
`import mcp_use` is deliberately NOT at module level (see git history/
PROTOTYPE.md for the incident this fixes): mcp_use/__init__.py eagerly
imports mcp_use.observability, which initializes Laminar/Langfuse tracing
integrations at import time, and that import has been observed to hang
indefinitely (blocked in a worker-thread `read()`, confirmed with `sample`
on the stuck process) even with no Laminar/Langfuse API keys configured --
i.e. even the branch that's supposed to no-op can stall. Since
src/graph/workflow.py imports this module eagerly (`from
src.agents.mcp_rdf_agent import run_mcp_agent`), an eager `import mcp_use`
here would take down the entire investigation service's startup (and
src/run.py's / eval's imports) over a CSKG-lookup dependency that most
questions never even touch. Deferred to inside get_mcp_client()/
run_mcp_agent() instead -- a hang there now only affects the one CSKG
question that triggers it (already wrapped in NODE_CALL_TIMEOUT_SECONDS by
src/graph/workflow.py's dispatch_retrieval_node), not process startup.
"""
import os
import logging
from pathlib import Path
from src.config.settings import llm

logger = logging.getLogger(__name__)

# --- Konfigurasi dan Inisialisasi Agen MCP ---

_mcp_client = None

def get_mcp_client():
    """Inisialisasi dan mengembalikan MCPClient (hanya sekali)."""
    global _mcp_client
    if _mcp_client is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "browser_mcp.json"
        if not config_path.exists():
            raise FileNotFoundError(f"MCP config file not found at {config_path}")
        # Must be set before `import mcp_use` -- its observability
        # submodules read these via os.getenv at their own import time
        # (see module docstring above), not lazily.
        os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"
        os.environ.setdefault("MCP_USE_LAMINAR", "false")
        os.environ.setdefault("MCP_USE_LANGFUSE", "false")
        from mcp_use import MCPClient
        _mcp_client = MCPClient.from_config_file(str(config_path))
    return _mcp_client

# Exact marker MAX_STEPS-exhausted/nothing-found responses must start
# with -- src/graph/workflow.py::_context_richness matches on this (plus a
# phrase-based heuristic fallback for whenever the model doesn't follow
# this instruction exactly) to tell "found nothing" apart from "found
# something," instead of the two being indistinguishable natural-language
# prose. Without this, a CSKG "I couldn't find anything" answer was being
# counted as substantive evidence, inflating the confidence cap
# _context_richness/_cap_confidence exist to enforce.
NO_DATA_MARKER = "NO_RELEVANT_DATA_FOUND"

strict_system_prompt = f"""
You are a specialized cybersecurity assistant. You MUST answer questions ONLY by using the provided tools. Your only source of information is the knowledge graph accessed via tools.

This knowledge base ONLY contains public reference data: MITRE ATT&CK techniques/tactics/groups/software/mitigations,
and CVE/CVSS records. It has NO IP addresses, hostnames, domains, live threat-intelligence feeds, IOC reputation data,
or any organization-specific data whatsoever. A question that asks you to correlate, attribute, or look up a
*specific* IP address, hostname, or piece of an organization's own environment against threat intelligence cannot be
answered here at all, no matter how the tools are used -- recognize this immediately from the question itself,
without attempting tool calls first, and respond per rule 6 below.

Most tools that take a technique/group/mitigation/software name (get_mitigations_for_technique,
get_techniques_by_tactic, get_techniques_used_by_group, etc.) match against that entity's TITLE TEXT
(e.g. "Network Service Discovery"), NOT its MITRE ID (e.g. "T1046") -- passing a bare ID to one of those
tools will silently return nothing, every time, no matter how many times you retry it, because the ID
never appears as a substring of the title. If a question names a technique only by ID: first call
get_technique_by_id to resolve it to its real name, THEN use that name with a name-matching tool -- OR use
text_to_sparql to write a single query that filters on the ID directly (FILTER(STRENDS(STR(?technique),
"/T1046"))), which sidesteps the name-matching requirement entirely.

Your thought process MUST be:
    1.  **Analyze the user's question** to understand the core intent.
    2.  **Select the best tool.**.
    3.  **Execute the tool.**  If that fails or the question is complex, use `text_to_sparql` to convert the question into a precise SPARQL query.
    4.  **Analyze the result.**
        - If the result is a **validation error** (like a Pydantic error), it means you provided the wrong arguments to the tool. Read the error message carefully. DO NOT use the same tool with the exact same arguments again. Correct the arguments and retry. For tools requiring `ctx`, DO NOT provide a value for it; the system handles it.
        - If the result is **empty or "not found"**, the information may not exist, or your query was too narrow. Try rephrasing your input for the tool, perhaps using a broader term.
        - If you are stuck in a loop of failures, only then you must state that you could not find the information.
    5.  **NEVER provide an answer from memory.** All answers must be based on tool results.
    6.  **If the question needs data this knowledge base doesn't have at all** (per the scope note above), OR **if,
        after reasonable attempts, no relevant information was found**, your final answer MUST begin with the exact
        marker "{NO_DATA_MARKER}" (nothing before it), followed by one short sentence saying why (out of scope, or
        what you tried). This applies even when you go on to offer a related but different reformulation of the
        question -- the marker reflects that the ORIGINAL question wasn't answered, regardless of what else you add
        after it. A downstream check relies on this exact marker to tell "found nothing" apart from "found
        something" -- do not paraphrase it, and do not include it if you DID find relevant information answering the
        question as actually asked.
"""

async def run_mcp_agent(question: str) -> str:
    """
    Runs the MCPAgent with the given question and returns the result.

    Uses agent.stream()/_consume_and_return() instead of the simpler
    agent.run() specifically to recover `tools_used_names` -- run() calls
    the exact same internal machinery but discards that information,
    returning only the final text. Without it there was no way to tell
    "the agent actually queried the knowledge graph and found nothing" apart
    from "the agent never called a single tool and just answered from its
    own reasoning" -- both can produce fluent, plausible-sounding prose (see
    NO_DATA_MARKER's own docstring for a real example of the latter slipping
    through undetected). This is a structural ground-truth signal instead of
    inferring intent from natural language.
    """
    try:
        client = get_mcp_client()  # imports mcp_use as a side effect -- see module docstring
        from mcp_use import MCPAgent
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=30,
            verbose=True,
            system_prompt=strict_system_prompt
        )
        generator = agent.stream(question, track_execution=False)
        result, steps_taken, tools_used_names = await agent._consume_and_return(generator)
        if tools_used_names:
            logger.info(f"[mcp_rdf_agent] question={question!r} steps={steps_taken} tools_used={tools_used_names}")
        else:
            logger.warning(
                f"[mcp_rdf_agent] question={question!r} answered WITHOUT calling any tool "
                f"(steps={steps_taken}) -- this is not a real knowledge-graph lookup, whatever the "
                f"response text claims"
            )
        return result
    except Exception as e:
        logger.error(f"An error occurred while running the MCP Agent: {e}")
        return f"Error during MCP agent execution: {e}"