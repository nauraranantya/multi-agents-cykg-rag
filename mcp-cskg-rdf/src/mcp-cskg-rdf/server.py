#!/usr/bin/env python3
"""
MITRE ATT&CK SPARQL MCP Server with Local/Remote Support

This server provides tools to query MITRE ATT&CK data using SPARQL queries
against either local RDF files or remote SPARQL endpoints based on the MITRE ATT&CK ontology.
"""

from typing import Any, Dict, List, Optional
import os
import argparse
import json
import socket
import sys
import time
import tiktoken
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# rdflib's SPARQLConnector.query() calls urllib's urlopen() with no timeout
# on any code path (verified directly in its source) -- a slow/stuck
# endpoint (e.g. the w3id.org -> http://sepses.ifs.tuwien.ac.at redirect
# chain measured taking 30s+ on some queries) hangs the request forever,
# freezing the whole agent loop with no recovery (observed directly: an
# 11+ minute hang on a single CallToolRequest). Passing timeout= to
# SPARQLStore's constructor doesn't help -- it's never extracted from
# self.kwargs and threaded through to urlopen(). Setting the process-wide
# default socket timeout is the only lever that actually reaches these
# calls; every socket operation in this process that doesn't set its own
# timeout will raise socket.timeout after this many seconds instead of
# blocking indefinitely.
socket.setdefaulttimeout(30)

# Some SPARQL endpoints (e.g. the public SEPSES endpoint at w3id.org) present
# a certificate chain that isn't in every installed certifi snapshot, even
# though it validates fine against the OS's own trust store (verified: curl
# and macOS both accept it). truststore patches ssl to verify against the OS
# trust store instead of certifi's bundled one -- must run before rdflib's
# SPARQLStore (via urllib) opens any HTTPS connection.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import rdflib
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base

# Configure logging
logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )

# Check for SPARQLStore availability
try:
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
    HAS_SPARQLSTORE = True
except ImportError:
    HAS_SPARQLSTORE = False
    logger.warning("SPARQLStore not available. SPARQL Endpoint Mode will be disabled.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MITRE ATT&CK SPARQL MCP Server v1.0.0")
parser.add_argument("--rdf-file", default="", help="Path to the local RDF file containing MITRE ATT&CK data")
parser.add_argument("--sparql-endpoint", default="", help="SPARQL endpoint URL (empty for Local File Mode)")
args = parser.parse_args()

logger.info("Starting MITRE ATT&CK SPARQL MCP Server v1.0.0")

# Define namespaces
ATTACK = rdflib.Namespace("http://w3id.org/sepses/vocab/ref/attack#")
CAPEC = rdflib.Namespace("http://w3id.org/sepses/vocab/ref/capec#")
RDF = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")

# Initialize FastMCP server
mcp = FastMCP(
    "MITRE ATT&CK SPARQL",
    dependencies=["rdflib[sparql]"],
    lifespan=lambda mcp: attack_triplestore_lifespan(mcp, args.rdf_file, args.sparql_endpoint)
)

@asynccontextmanager
async def attack_triplestore_lifespan(server: FastMCP, rdf_file: str, sparql_endpoint: str) -> AsyncIterator[Dict[str, Any]]:
    """Manage the lifespan of the MITRE ATT&CK triplestore.

    Args:
        server (FastMCP): The FastMCP server instance.
        rdf_file (str): Path to the local RDF file.
        sparql_endpoint (str): URL of the SPARQL endpoint, if any.

    Yields:
        Dict[str, Any]: Context dictionary containing the graph and configuration.
    """
    logger.info(f"Initializing MITRE ATT&CK triplestore with rdf_file={rdf_file}, sparql_endpoint={sparql_endpoint}")
    
    metrics = {"queries": 0, "total_time": 0.0}
    max_tokens = 10000
    
    if sparql_endpoint and HAS_SPARQLSTORE:
        logger.info(f"Connecting to SPARQL endpoint: {sparql_endpoint}")
        try:
            graph = SPARQLStore(query_endpoint=sparql_endpoint)
            # Test connection
            graph.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
            logger.info(f"Successfully connected to {sparql_endpoint}")
        except Exception as e:
            logger.error(f"Failed to connect to SPARQL endpoint: {str(e)}")
            raise
    else:
        graph = rdflib.Graph()
        file_path = os.path.join(os.path.dirname(__file__), rdf_file)
        logger.info(f"Loading local RDF file: {file_path}")
        try:
            graph.parse(file_path, format="turtle")
            logger.info(f"Loaded {len(graph)} triples from local file")
        except FileNotFoundError:
            logger.error(f"RDF file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load RDF file: {str(e)}")
            raise    
    try:
        logger.info("MITRE ATT&CK triplestore initialized successfully")
        yield {
            "graph": graph,
            "metrics": metrics,
            "max_tokens": max_tokens,
            "rdf_file": rdf_file,
            "sparql_endpoint": sparql_endpoint,
            "is_sparql_endpoint": bool(sparql_endpoint and HAS_SPARQLSTORE)
        }
    finally:
        logger.info("Shutting down MITRE ATT&CK triplestore connection")
        if sparql_endpoint and HAS_SPARQLSTORE:
            try:
                graph.close()
            except:
                pass

MAX_DESCRIPTION_CHARS = 400
MAX_RESULT_CHARS = 6000  # ~1500 tokens -- keeps a single tool call's context contribution bounded


def format_sparql_results(results, include_description: bool = False) -> str:
    """Format SPARQL query results into a readable string.

    Bounded on two axes, independent of whether the underlying SPARQL query
    itself has a LIMIT (many of the ~40 tools don't): each `description`
    field is truncated to MAX_DESCRIPTION_CHARS, and the total formatted
    output is truncated to MAX_RESULT_CHARS (by whole result rows, never
    mid-row) with a note about how many rows were omitted. Without this, a
    broad keyword/tactic match can return dozens of full-paragraph
    descriptions in one tool call, ballooning the agent's context and
    latency for no benefit -- observed directly, not just a theoretical risk.

    Args:
        results: SPARQL query results
        include_description: Whether to include descriptions (if available)

    Returns:
        Formatted string representation of the results
    """
    if not results:
        return "No results found."

    formatted_results = []

    for row in results:
        result_parts = []

        # Handle different variable bindings
        for var_name, value in row.asdict().items():
            if value:
                # Clean up the value representation
                if isinstance(value, rdflib.URIRef):
                    # Extract the local name for cleaner display
                    local_name = str(value).split('#')[-1] if '#' in str(value) else str(value).split('/')[-1]
                    result_parts.append(f"{var_name}: {local_name}")
                else:
                    text = str(value)
                    if var_name == "description" and len(text) > MAX_DESCRIPTION_CHARS:
                        text = text[:MAX_DESCRIPTION_CHARS].rstrip() + "... [truncated]"
                    result_parts.append(f"{var_name}: {text}")

        formatted_results.append(" | ".join(result_parts))

    output = "\n".join(formatted_results)
    if len(output) > MAX_RESULT_CHARS:
        kept, running = [], 0
        for line in formatted_results:
            if running + len(line) + 1 > MAX_RESULT_CHARS:
                break
            kept.append(line)
            running += len(line) + 1
        omitted = len(formatted_results) - len(kept)
        output = "\n".join(kept) + (
            f"\n... [{omitted} more result(s) truncated -- narrow your query "
            f"(e.g. a more specific keyword/tactic, or include_description=False) to see more]"
        )
    return output

#################################################################
# Core Infrastructure Tools
#################################################################
# Tools
@mcp.tool()
def set_max_tokens(tokens: int, ctx: Context) -> str:
    """Set the maximum token limit for prompts.

    Args:
        tokens (int): The new maximum token limit (must be positive).
        ctx (Context): The FastMCP context object.

    Returns:
        str: Confirmation message or error if the value is invalid.
    """
    if tokens <= 0:
        return "Error: MAX_TOKENS must be positive."
    ctx.request_context.lifespan_context["max_tokens"] = tokens
    logger.info(f"Set MAX_TOKENS to {tokens}")
    return f"MAX_TOKENS set to {tokens}"

@mcp.tool()
def execute_sparql_query(query: str, ctx: Context, include_description: bool = False) -> str:
    """Execute a custom SPARQL query against the MITRE ATT&CK knowledge graph.
    
    Args:
        query: SPARQL query string to execute
        ctx: FastMCP context object
        include_description: Whether to include descriptions in results (default: False)
        
    Returns:
        Formatted query results
    """
    graph = ctx.request_context.lifespan_context["graph"]
    start_time = time.time()
    
    try:
        results = graph.query(query)
        ctx.request_context.lifespan_context["metrics"]["queries"] += 1
        ctx.request_context.lifespan_context["metrics"]["total_time"] += time.time() - start_time
        logger.info(query)
        return format_sparql_results(results, include_description)
    except Exception as e:
        logger.error(f"SPARQL query error: {str(e)}")
        return f"Error executing SPARQL query: {str(e)}"

@mcp.tool()
def get_server_mode(ctx: Context) -> str:
    """Get the current mode of the MITRE ATT&CK server.
    
    Args:
        ctx: FastMCP context object
        
    Returns:
        A message indicating the mode and data source
    """
    rdf_file = ctx.request_context.lifespan_context["rdf_file"]
    sparql_endpoint = ctx.request_context.lifespan_context["sparql_endpoint"]
    is_sparql_endpoint = ctx.request_context.lifespan_context["is_sparql_endpoint"]
    
    if is_sparql_endpoint:
        return f"SPARQL Endpoint Mode with Endpoint: '{sparql_endpoint}'"
    else:
        return f"Local File Mode with Dataset: '{rdf_file or 'empty graph'}'"

@mcp.tool()
def get_attack_statistics(ctx: Context) -> str:
    """Get statistical summary of the MITRE ATT&CK knowledge base.
    
    Args:
        ctx: FastMCP context object
        
    Returns:
        JSON string containing statistics about the knowledge base
    """
    graph = ctx.request_context.lifespan_context["graph"]
    is_sparql_endpoint = ctx.request_context.lifespan_context["is_sparql_endpoint"]
    
    try:
        if is_sparql_endpoint:
            # For SPARQL endpoints, use limited queries to avoid timeouts
            query = """
            PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
            SELECT 
                (COUNT(DISTINCT ?technique) AS ?techniqueCount)
                (COUNT(DISTINCT ?group) AS ?groupCount)
                (COUNT(DISTINCT ?software) AS ?softwareCount)
                (COUNT(DISTINCT ?mitigation) AS ?mitigationCount)
                (COUNT(DISTINCT ?tactic) AS ?tacticCount)
            WHERE {
                OPTIONAL { ?technique a attack:Technique }
                OPTIONAL { ?group a attack:AdversaryGroup }
                OPTIONAL { ?software a attack:Software }
                OPTIONAL { ?mitigation a attack:Mitigation }
                OPTIONAL { ?tactic a attack:Tactic }
            }
            """
        else:
            # For local graphs, we can do more comprehensive statistics
            query = """
            PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
            SELECT 
                (COUNT(DISTINCT ?technique) AS ?techniqueCount)
                (COUNT(DISTINCT ?subtechnique) AS ?subtechniqueCount)
                (COUNT(DISTINCT ?group) AS ?groupCount)
                (COUNT(DISTINCT ?software) AS ?softwareCount)
                (COUNT(DISTINCT ?malware) AS ?malwareCount)
                (COUNT(DISTINCT ?mitigation) AS ?mitigationCount)
                (COUNT(DISTINCT ?tactic) AS ?tacticCount)
                (COUNT(DISTINCT ?asset) AS ?assetCount)
                (COUNT(DISTINCT ?dataSource) AS ?dataSourceCount)
                (COUNT(DISTINCT ?dataComponent) AS ?dataComponentCount)
            WHERE {
                OPTIONAL { ?technique a attack:Technique }
                OPTIONAL { ?subtechnique a attack:SubTechnique }
                OPTIONAL { ?group a attack:AdversaryGroup }
                OPTIONAL { ?software a attack:Software }
                OPTIONAL { ?malware a attack:Malware }
                OPTIONAL { ?mitigation a attack:Mitigation }
                OPTIONAL { ?tactic a attack:Tactic }
                OPTIONAL { ?asset a attack:Asset }
                OPTIONAL { ?dataSource a attack:DataSource }
                OPTIONAL { ?dataComponent a attack:DataComponent }
            }
            """
        
        results = graph.query(query)
        stats = {}
        for row in results:
            for var_name, value in row.asdict().items():
                if value is not None:
                    stats[var_name] = int(value)
        
        return json.dumps(stats, indent=2)
    except Exception as e:
        logger.error(f"Statistics query error: {str(e)}")
        return f"Error retrieving statistics: {str(e)}"

@mcp.tool()
def health_check(ctx: Context) -> str:
    """Check the health of the MITRE ATT&CK triplestore connection.
    
    Args:
        ctx: FastMCP context object
        
    Returns:
        Health status message
    """
    graph = ctx.request_context.lifespan_context["graph"]
    try:
        # Simple test query
        results = list(graph.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"))
        return "Healthy - MITRE ATT&CK triplestore is responsive"
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return f"Unhealthy: {str(e)}"

#################################################################
# Technique Query Tools
#################################################################

@mcp.tool()
def get_all_techniques(ctx: Context,  include_description: bool = False) -> str:
    """Get all techniques in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?label ?description WHERE {
        ?technique a attack:Technique .
        ?technique dcterm:title ?label .
        OPTIONAL { ?technique dcterm:description ?description }  
    }
    ORDER BY ?label
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_by_keyword(ctx: Context,  keyword: str, include_description: bool = False) -> str:
    """Get all techniques in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?label ?description WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?label .
        OPTIONAL {{ ?technique dcterm:description ?description }}  
        
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}")) ||
            CONTAINS(LCASE(?description), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    LIMIT 50
    """
    return execute_sparql_query(query, ctx, include_description)


@mcp.tool()
def get_technique_by_id(technique_id: str, ctx: Context, include_description: bool = True) -> str:
    """Get a specific MITRE ATT&CK technique or sub-technique by its ATT&CK ID.

    Use this when the question references a technique by ID (e.g. "What is
    T1505.003?") rather than by name -- get_techniques_by_keyword and the
    other technique tools only match against the technique's title/description
    text, not its ID, so an ID-only question returns no results there even
    though the technique exists.

    Args:
        technique_id: ATT&CK technique ID, e.g. 'T1505.003' or 'T1110'
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: True)
    """
    normalized_id = technique_id.strip().upper()
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?technique ?label ?description ?tacticLabel WHERE {{
        ?technique a attack:Technique .
        FILTER(STRENDS(STR(?technique), "/{normalized_id}"))
        ?technique dcterm:title ?label .
        OPTIONAL {{ ?technique dcterm:description ?description }}
        OPTIONAL {{
            ?technique attack:accomplishesTactic ?tactic .
            ?tactic dcterm:title ?tacticLabel .
        }}
    }}
    """
    return execute_sparql_query(query, ctx, include_description)


@mcp.tool()
def get_techniques_by_tactic(tactic_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques that accomplish a specific tactic.
    
    Args:
        tactic_name: Name of the tactic to search for
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?tactic ?tacticLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:accomplishesTactic ?tactic .
        ?tactic dcterm:title ?tacticLabel .
        FILTER(CONTAINS(LCASE(?tacticLabel), LCASE("{tactic_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_subtechniques_of_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all subtechniques of a parent technique.
    
    Args:
        technique_name: Name of the parent technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?subtechnique ?subtechniqueLabel ?parentTechnique ?parentLabel WHERE {{
        ?subtechnique a attack:SubTechnique .
        ?subtechnique dcterm:title ?subtechniqueLabel .
        ?subtechnique attack:isSubTechniqueOf ?parentTechnique .
        ?parentTechnique dcterm:title ?parentLabel .
        FILTER(CONTAINS(LCASE(?parentLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?subtechniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_by_platform(platform: str, ctx: Context, include_description: bool = False) -> str:
    """Get techniques that target a specific platform.
    
    Args:
        platform: Platform name (e.g., Windows, Linux, macOS)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    # ?platform is bound to a URI resource (e.g. .../attack/platform/Linux),
    # not a string literal -- LCASE(?platform) directly errors on Virtuoso
    # ("SL001: LCASE() needs a string value as 1st argument"). STR() first
    # converts it to its string form (the platform name survives as the
    # URI's last path segment, so CONTAINS still matches correctly).
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>

    SELECT ?technique ?label ?platform WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?label .
        ?technique attack:platform ?platform .
        FILTER(CONTAINS(LCASE(STR(?platform)), LCASE("{platform}")))
    }}
    ORDER BY ?label
    """

    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Adversary Group Query Tools
#################################################################

@mcp.tool()
def get_all_adversary_groups(ctx: Context, include_description: bool = False) -> str:
    """Get all adversary groups in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?label ?aliases WHERE {
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?label .
        OPTIONAL { ?group attack:aliases ?aliases }
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_used_by_group(group_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques used by a specific adversary group.
    
    Args:
        group_name: Name of the adversary group
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?technique ?techniqueLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        ?group attack:usesTechnique ?technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?groupLabel), LCASE("{group_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_software_used_by_group(group_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all software used by a specific adversary group.
    
    Args:
        group_name: Name of the adversary group
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?software ?softwareLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        {{
            ?group attack:usesSoftware ?software .
        }} UNION {{
            ?group attack:usesMalware ?software .
        }}
        ?software dcterm:title ?softwareLabel .
        FILTER(CONTAINS(LCASE(?groupLabel), LCASE("{group_name}")))
    }}
    ORDER BY ?softwareLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_groups_using_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all adversary groups that use a specific technique.
    
    Args:
        technique_name: Name of the technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?technique ?techniqueLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        ?group attack:usesTechnique ?technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?groupLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Software and Malware Query Tools
#################################################################

@mcp.tool()
def get_all_software(ctx: Context, include_description: bool = False) -> str:
    """Get all software in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?software ?label ?type WHERE {
        ?software a ?type .
        ?software dcterm:title ?label .
        FILTER(?type = attack:Software || ?type = attack:Malware)
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_software_by_keyword(ctx: Context, keyword: str, include_description: bool = False) -> str:
    """Get all software in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?software ?label ?type WHERE {{
        ?software a ?type .
        ?software dcterm:title ?label .
        FILTER(?type = attack:Software || ?type = attack:Malware)
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()

def get_techniques_used_by_software(software_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques implemented by specific software/malware.
    
    Args:
        software_name: Name of the software or malware
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?software ?softwareLabel ?technique ?techniqueLabel WHERE {{
        {{
            ?software a attack:Software .
            ?software dcterm:title ?softwareLabel .
            ?technique attack:hasSoftware ?software .
        }} UNION {{
            ?software a attack:Malware .
            ?software dcterm:title ?softwareLabel .
            ?software attack:implementsTechnique ?technique .
        }}
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?softwareLabel), LCASE("{software_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Mitigation Query Tools
#################################################################

@mcp.tool()
def get_all_mitigations(ctx: Context, include_description: bool = False) -> str:
    """Get all mitigations in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?mitigation ?label WHERE {
        ?mitigation a attack:Mitigation .
        ?mitigation dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

def get_all_mitigations_by_keyword(ctx: Context, keyword: str, include_description: bool = False) -> str:
    """Get all mitigations in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?mitigation ?label WHERE {{
        ?mitigation a attack:Mitigation .
        ?mitigation dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_mitigated_by_mitigation(mitigation_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques that are mitigated by a specific mitigation.
    
    Args:
        mitigation_name: Name of the mitigation
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?mitigation ?mitigationLabel ?technique ?techniqueLabel WHERE {{
        ?mitigation a attack:Mitigation .
        ?mitigation dcterm:title ?mitigationLabel .
        ?mitigation attack:preventsTechnique ?technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?mitigationLabel), LCASE("{mitigation_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_mitigations_for_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all mitigations that can prevent a specific technique.
    
    Args:
        technique_name: Name of the technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    # The underlying CSKG has duplicate RDF triples and parallel canonical
    # (T1078) / slug (valid-accounts) URIs for the same real technique or
    # mitigation -- selecting the raw ?technique/?mitigation URIs multiplies
    # those duplicates together in the join, producing dozens of rows that
    # are semantically identical. The URIs aren't informative to read anyway
    # (e.g. "course-of-action--f9f9e6ef-..."), so project + DISTINCT on just
    # the human-readable labels instead, which collapses both duplication
    # sources at once.
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?techniqueLabel ?mitigationLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:hasMitigation ?mitigation .
        ?mitigation dcterm:title ?mitigationLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?mitigationLabel
    """

    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Tactic Query Tools
#################################################################

@mcp.tool()
def get_all_tactics(ctx: Context, include_description: bool = False) -> str:
    """Get all tactics in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?tactic ?label WHERE {
        ?tactic a attack:Tactic .
        ?tactic dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_tactics_by_keyword(ctx: Context, keyword:str, include_description: bool = False) -> str:
    """Get all tactics in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?tactic ?label WHERE {{
        ?tactic a attack:Tactic .
        ?tactic dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_tactics_for_technique(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all tactics accomplished by a specific technique.
    
    Args:
        technique_name: Name of the technique
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?tactic ?tacticLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:accomplishesTactic ?tactic .
        ?tactic dcterm:title ?tacticLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
    }}
    ORDER BY ?tacticLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Asset Query Tools (for ICS)
#################################################################

@mcp.tool()
def get_all_assets(ctx: Context, include_description: bool = False) -> str:
    """Get all assets in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?asset ?label WHERE {
        ?asset a attack:Asset .
        ?asset dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_assets_by_keyword(ctx: Context, keyword:str, include_description: bool = False) -> str:
    """Get all assets in the MITRE ATT&CK framework.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?asset ?label WHERE {{
        ?asset a attack:Asset .
        ?asset dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}")) ||
            CONTAINS(LCASE(?description), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_techniques_targeting_asset(asset_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get all techniques that target a specific asset.
    
    Args:
        asset_name: Name of the asset
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?asset ?assetLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        ?technique attack:targetsAsset ?asset .
        ?asset dcterm:title ?assetLabel .
        FILTER(CONTAINS(LCASE(?assetLabel), LCASE("{asset_name}")))
    }}
    ORDER BY ?techniqueLabel
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Data Source and Component Query Tools
#################################################################

@mcp.tool()
async def get_all_data_sources(ctx: Context, include_description: bool = False) -> str:
    """Get all data sources in the MITRE ATT&CK framework.
    
    Args:
        include_description: Whether to include descriptions (default: False)
    """
     
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?dataSource ?label WHERE {
        ?dataSource a attack:DataSource .
        ?dataSource dcterm:title ?label .
    }
    ORDER BY ?label
    """
    
    return format_sparql_results(query, ctx, include_description)

@mcp.tool()
async def get_data_sources_by_keyword(ctx: Context, keyword:str, include_description: bool = False) -> str:
    """Get all data sources in the MITRE ATT&CK framework.
    
    Args:
        include_description: Whether to include descriptions (default: False)
    """
     
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?dataSource ?label WHERE {{
        ?dataSource a attack:DataSource .
        ?dataSource dcterm:title ?label .
        FILTER(
            CONTAINS(LCASE(?label), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?label
    """
    
    return format_sparql_results(query, ctx, include_description)

@mcp.tool()
async def get_all_data_components(ctx: Context, include_description: bool = False) -> str:
    """Get all data components in the MITRE ATT&CK framework.
    
    Args:
        include_description: Whether to include descriptions (default: False)
    """    
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?dataComponent ?label WHERE {
        ?dataComponent a attack:DataComponent .
        ?dataComponent dcterm:title ?label .
    }
    ORDER BY ?label
    """
    

    return format_sparql_results(query, ctx, include_description)

#################################################################
# Complex Relationship Queries
#################################################################

@mcp.tool()
async def get_technique_relationships(technique_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get comprehensive relationships for a specific technique.
    
    Args:
        technique_name: Name of the technique
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?technique ?techniqueLabel ?relationshipType ?relatedEntity ?relatedLabel WHERE {{
        ?technique a attack:Technique .
        ?technique dcterm:title ?techniqueLabel .
        FILTER(CONTAINS(LCASE(?techniqueLabel), LCASE("{technique_name}")))
        
        {{
            ?technique attack:accomplishesTactic ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("accomplishes_tactic" AS ?relationshipType)
        }} UNION {{
            ?technique attack:hasMitigation ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("has_mitigation" AS ?relationshipType)
        }} UNION {{
            ?technique attack:hasSoftware ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("has_software" AS ?relationshipType)
        }} UNION {{
            ?relatedEntity attack:usesTechnique ?technique .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("used_by_group" AS ?relationshipType)
        }} UNION {{
            ?technique attack:targetsAsset ?relatedEntity .
            ?relatedEntity dcterm:title ?relatedLabel .
            BIND("targets_asset" AS ?relationshipType)
        }}
    }}
    ORDER BY ?relationshipType ?relatedLabel
    """
    return format_sparql_results(query, ctx, include_description)

@mcp.tool()
async def get_group_capabilities(group_name: str, ctx: Context, include_description: bool = False) -> str:
    """Get comprehensive capabilities (techniques, software, malware) for an adversary group.
    
    Args:
        group_name: Name of the adversary group
        include_description: Whether to include descriptions (default: False)
    """
   
    query = f"""
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    PREFIX dcterm: <http://purl.org/dc/terms/>
    
    SELECT ?group ?groupLabel ?capabilityType ?capability ?capabilityLabel WHERE {{
        ?group a attack:AdversaryGroup .
        ?group dcterm:title ?groupLabel .
        FILTER(CONTAINS(LCASE(?groupLabel), LCASE("{group_name}")))
        
        {{
            ?group attack:usesTechnique ?capability .
            ?capability dcterm:title ?capabilityLabel .
            BIND("technique" AS ?capabilityType)
        }} UNION {{
            ?group attack:usesSoftware ?capability .
            ?capability dcterm:title ?capabilityLabel .
            BIND("software" AS ?capabilityType)
        }} UNION {{
            ?group attack:usesMalware ?capability .
            ?capability dcterm:title ?capabilityLabel .
            BIND("malware" AS ?capabilityType)
        }}
    }}
    ORDER BY ?capabilityType ?capabilityLabel
    """
    return format_sparql_results(query, ctx, include_description)

#################################################################
# Statistics and Summary Tools
#################################################################

@mcp.tool()
async def get_attack_statistics() -> str:
    """Get statistical summary of the MITRE ATT&CK knowledge base."""
    
    query = """
    PREFIX attack: <http://w3id.org/sepses/vocab/ref/attack#>
    
    SELECT 
        (COUNT(DISTINCT ?technique) AS ?techniqueCount)
        (COUNT(DISTINCT ?subtechnique) AS ?subtechniqueCount)
        (COUNT(DISTINCT ?group) AS ?groupCount)
        (COUNT(DISTINCT ?software) AS ?softwareCount)
        (COUNT(DISTINCT ?malware) AS ?malwareCount)
        (COUNT(DISTINCT ?mitigation) AS ?mitigationCount)
        (COUNT(DISTINCT ?tactic) AS ?tacticCount)
        (COUNT(DISTINCT ?asset) AS ?assetCount)
    WHERE {
        OPTIONAL { ?technique a attack:Technique }
        OPTIONAL { ?subtechnique a attack:SubTechnique }
        OPTIONAL { ?group a attack:AdversaryGroup }
        OPTIONAL { ?software a attack:Software }
        OPTIONAL { ?malware a attack:Malware }
        OPTIONAL { ?mitigation a attack:Mitigation }
        OPTIONAL { ?tactic a attack:Tactic }
        OPTIONAL { ?asset a attack:Asset }
    }
    """
    return format_sparql_results(query)

#################################################################
# CVE Query Tools
#################################################################

@mcp.tool()
def get_all_cves(ctx: Context, include_description: bool = False) -> str:
    """Get all CVEs in the knowledge base.
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    
    SELECT ?cve ?description WHERE {
        ?cve a cve:CVE .
        OPTIONAL { ?cve dcterms:description ?description }
    }
    ORDER BY ?cve
    LIMIT 50
    """
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_cve_by_id(cve_id: str, ctx: Context, include_description: bool = False) -> str:
    """Get detailed information about a specific CVE.
    
    Args:
        cve_id: CVE identifier (e.g., CVE-2023-1234)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    
    SELECT ?cve ?description ?publishedDate ?modifiedDate WHERE {{
        ?cve a cve:CVE .
        FILTER(CONTAINS(STR(?cve), "{cve_id}"))
        OPTIONAL {{ ?cve dcterms:description ?description }}
        OPTIONAL {{ ?cve dcterms:created ?publishedDate }}
        OPTIONAL {{ ?cve dcterms:modified ?modifiedDate }}
    }}
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def search_cves_by_keyword(keyword: str, ctx: Context, include_description: bool = False) -> str:
    """Search CVEs by keyword in title or description.
    
    Args:
        keyword: Keyword to search for
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    
    SELECT ?cve ?description WHERE {{
        ?cve a cve:CVE .
        OPTIONAL {{ ?cve dcterms:description ?description }}
        FILTER(
            CONTAINS(LCASE(?description), LCASE("{keyword}"))
        )
    }}
    ORDER BY ?cve
    LIMIT 50
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# CVSS Query Tools
#################################################################

@mcp.tool()
def get_cves_by_cvss_score(min_score: float, max_score: float, ctx:Context, include_description: bool = False) -> str:
    """Get CVEs within a specific CVSS score range.
    
    Args:
        min_score: Minimum CVSS score
        max_score: Maximum CVSS score
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
        PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
        PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?cve ?description ?baseScore WHERE {{
            ?cve a cve:CVE .
            ?cve cve:hasCVSS3BaseMetric ?cvss3 .
            ?cvss3 cvss:baseScore ?baseScore .
            OPTIONAL {{ ?cve dcterms:description ?description }}
            FILTER(xsd:integer(?baseScore) >= {min_score} && xsd:integer(?baseScore) <= {max_score})
        }}
        ORDER BY DESC(?baseScore)
        """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_high_severity_cves(ctx: Context, include_description: bool = False) -> str:
    """Get CVEs with high severity (CVSS score >= 7.0).
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    
    SELECT ?cve ?description ?baseScore WHERE {
        ?cve a cve:CVE .
            ?cve cve:hasCVSS3BaseMetric ?cvss3 .
            ?cvss3 cvss:baseScore ?baseScore .
        OPTIONAL { ?cve dcterms:description ?description }
        FILTER(xsd:integer(?baseScore) >= 7.0)
    }
    ORDER BY DESC(?baseScore)
    LIMIT 50
    """
    
    return execute_sparql_query(query, ctx, include_description)

@mcp.tool()
def get_critical_cves(ctx: Context, include_description: bool = False) -> str:
    """Get CVEs with critical severity (CVSS score >= 9.0).
    
    Args:
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = """
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    
    SELECT ?cve ?description ?baseScore WHERE {
        ?cve a cve:CVE .
            ?cve cve:hasCVSS3BaseMetric ?cvss3 .
            ?cvss3 cvss:baseScore ?baseScore .
        OPTIONAL { ?cve dcterms:description ?description }
        FILTER(xsd:integer(?baseScore) >= 9.0)
    }
    ORDER BY DESC(?baseScore)
    LIMIT 50
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Reference Query Tools
#################################################################

@mcp.tool()
def get_references_for_cve(cve_id: str, ctx: Context, include_description: bool = False) -> str:
    """Get all references for a specific CVE.
    
    Args:
        cve_id: CVE identifier (e.g., CVE-2023-1234)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    
    SELECT ?cve ?reference ?referenceUrl ?referenceSource ?referenceType WHERE {{
        ?cve a cve:CVE .
        ?cve cve:hasReference ?reference .
        FILTER(CONTAINS(STR(?cve), "{cve_id}"))
        OPTIONAL {{ ?reference cve:referenceUrl ?referenceUrl }}
        OPTIONAL {{ ?reference cve:referenceSource ?referenceSource }}
        OPTIONAL {{ ?reference cve:referenceType ?referenceType }}
    }}
    ORDER BY ?reference
    """
    
    return execute_sparql_query(query, ctx, include_description)

#################################################################
# Time-based Query Tools
#################################################################

@mcp.tool()
def get_recent_cves(days: int = 30, include_description: bool = False) -> str:
    """Get CVEs published in the last N days.
    
    Args:
        days: Number of days to look back (default: 30)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
    
    SELECT ?cve ?title ?publishedDate ?baseScore WHERE {{
        ?cve a cve:CVE .
        ?cve dcterms:created ?publishedDate .
        OPTIONAL {{ ?cve dcterms:title ?title }}
        
        OPTIONAL {{
            {{
                ?cve cve:hasCVSS3BaseMetric ?cvss3 .
                ?cvss3 cvss:baseScore ?baseScore .
            }} UNION {{
                ?cve cve:hasCVSS2BaseMetric ?cvss2 .
                ?cvss2 cvss:baseScore ?baseScore .
            }}
        }}
        
        FILTER(?publishedDate >= (NOW() - "P{days}D"^^xsd:duration))
    }}
    ORDER BY DESC(?publishedDate) DESC(?baseScore)
    LIMIT 100
    """
    
    return execute_sparql_query(query, include_description)

@mcp.tool()
def get_cves_by_year(year: int, ctx: Context, include_description: bool = False) -> str:
    """Get CVEs published in a specific year.
    
    Args:
        year: Year to filter by (e.g., 2023)
        ctx: FastMCP context object
        include_description: Whether to include descriptions (default: False)
    """
    query = f"""
    PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
    
    SELECT ?cve ?title ?publishedDate ?baseScore WHERE {{
        ?cve a cve:CVE .
        ?cve dcterms:created ?publishedDate .
        OPTIONAL {{ ?cve dcterms:title ?title }}
        
        OPTIONAL {{
            {{
                ?cve cve:hasCVSS3BaseMetric ?cvss3 .
                ?cvss3 cvss:baseScore ?baseScore .
            }} UNION {{
                ?cve cve:hasCVSS2BaseMetric ?cvss2 .
                ?cvss2 cvss:baseScore ?baseScore .
            }}
        }}
        
        FILTER(YEAR(?publishedDate) = {year})
    }}
    ORDER BY DESC(?publishedDate) DESC(?baseScore)
    LIMIT 500
    """
    
    return execute_sparql_query(query, ctx, include_description)

# Run the server
if __name__ == "__main__":
    logger.info("Starting mcp.run()")
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start RDF Explorer: {str(e)}")
        sys.exit(1)
    logger.info("mcp.run() completed")

@mcp.prompt()
def text_to_sparql(prompt: str, ctx: Context) -> str:
    """Convert a text prompt to a SPARQL query and execute it, with token limit checks.

    Args:
        prompt (str): The text prompt to convert to SPARQL.
        ctx (Context): The FastMCP context object.

    Returns:
        str: Query results with usage stats, or an error message if execution fails or token limits are exceeded.
    """
    encoder = tiktoken.get_encoding("gpt2")
    start_time = time.time()
    grok_response = {"endpoint": None, "query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"}  # Placeholder
    endpoint = grok_response.get("endpoint")
    query = grok_response["query"]
    logger.debug(f"Prompt received: {prompt}")
    input_tokens = len(encoder.encode(prompt + query))
    max_tokens = ctx.request_context.lifespan_context["max_tokens"]
    if input_tokens > max_tokens:
        logger.debug(f"Token limit exceeded: {input_tokens} > {max_tokens}")
        return f"Error: Input exceeds token limit ({input_tokens} tokens > {max_tokens}). Shorten your prompt or increase MAX_TOKENS with 'set_max_tokens'."
    active_endpoint = ctx.request_context.lifespan_context["active_external_endpoint"]
    use_local = active_endpoint is None and endpoint is None
    use_configured = active_endpoint and (endpoint is None or endpoint == active_endpoint)
    use_extracted = endpoint and endpoint != active_endpoint
    logger.debug(f"Execution context - Local: {use_local}, Configured: {use_configured}, Extracted: {use_extracted}")
    try:
        if use_extracted:
            results = ctx.request_context.call_tool("execute_on_endpoint", {"endpoint": endpoint, "query": query})
            logger.debug(f"Executed on extracted endpoint {endpoint}")
        elif use_local:
            results = ctx.request_context.call_tool("sparql_query", {"query": query, "use_service": False})
            logger.debug("Executed on local graph")
        elif use_configured:
            results = ctx.request_context.call_tool("sparql_query", {"query": query})
            logger.debug(f"Executed on configured endpoint {active_endpoint}")
        else:
            logger.debug("No valid execution context")
            return "Unable to determine execution context for the query."
        output_tokens = len(encoder.encode(results))
        total_tokens = input_tokens + output_tokens
        exec_time = time.time() - start_time
        usage_stats = f"[Resource Usage: Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total: {total_tokens}, Time: {exec_time:.2f}s]"
        logger.debug(f"Usage stats generated: {usage_stats}")
        return f"{results}\n\n{usage_stats}"
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        if "interrupted" in str(e).lower():
            return f"Error: Response interrupted, likely due to token limit (Input: {input_tokens} tokens, Max: {max_tokens}). Shorten input or increase MAX_TOKENS."
        return f"Error executing query: {str(e)}"