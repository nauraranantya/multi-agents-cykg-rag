# src/utils/logging_config.py
import os
import logging

def setup_logging():
    """Mengatur konfigurasi logging untuk proyek."""
    log_dir = "../log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Menghapus handler yang ada untuk menghindari duplikasi log
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'multi_agent_cykg.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Quiet third-party chatter that drowns out src/graph/workflow.py's
    # per-agent output lines (each node logs under its own "agent.<name>"
    # logger -- see workflow.py -- specifically so a reader can tell which
    # agent said what; a "HTTP Request: POST ... 200 OK" line from httpx
    # after every single LLM call, or a Neo4j "UnknownRelationshipType"
    # notification for every alert missing an optional field, defeats that
    # by burying the signal). Real errors from either still surface -- only
    # their routine INFO/WARNING noise is silenced.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)