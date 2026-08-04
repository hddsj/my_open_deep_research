"""Test script to verify the researcher subgraph works correctly."""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from my_deep_research.deep_researcher import researcher_subgraph

load_dotenv()


async def main():
    print("=" * 60)
    print("Testing researcher subgraph with a simple research topic")
    print("=" * 60)

    result = await researcher_subgraph.ainvoke(
        {
            "researcher_messages": [
                HumanMessage(content="Research the key features and architecture of LangGraph framework")
            ],
            "research_topic": "LangGraph framework features and architecture",
            "tool_call_iterations": 0,
        },
    )

    print(f"\nResult keys: {list(result.keys())}")
    print(f"\nCompressed research (first 500 chars):")
    print(result.get("compressed_research", "N/A")[:500])
    print(f"\n... (total length: {len(result.get('compressed_research', ''))} chars)")


asyncio.run(main())
