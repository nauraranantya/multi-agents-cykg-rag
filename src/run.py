# src/run.py
import argparse
import asyncio
from src.utils.logging_config import setup_logging
from src.graph.workflow import app
from src.retrieval.temporal import now_iso
import logging

async def main():
    """The main function is to run the agent."""
    setup_logging()
    logging.getLogger('mcp_use').propagate = False
    
    parser = argparse.ArgumentParser(description="Run Multi-Agent with questions.")
    parser.add_argument("question", type=str, help="Questions to ask agents.")
    args = parser.parse_args()

    initial_state = {
        "question": args.question,
        "original_question": args.question,
        "messages": [("human", args.question)],
        "query_timestamp": now_iso(),
    }

    config = {"recursion_limit": 30}

    final_result = await app.ainvoke(initial_state, config=config)
    
    print("\n--- Final Answer ---")
    print(final_result.get('answer'))

if __name__ == "__main__":
    asyncio.run(main())