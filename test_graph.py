"""Test script to verify the first LangGraph workflow runs correctly."""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from my_deep_research.deep_researcher import deep_researcher

load_dotenv()


async def main():
    print("=" * 60)
    print("Test 1: Clear question (should NOT ask for clarification)")
    print("=" * 60)

    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content="Write a report about the history and key features of LangGraph framework")]},
    )

    print(f"\nFinal state keys: {list(result.keys())}")
    print(f"Research brief: {result.get('research_brief', 'N/A')}")
    print(f"\nMessages ({len(result['messages'])}):")
    for msg in result["messages"]:
        print(f"  [{msg.__class__.__name__}] {msg.content[:150]}...")

    print("\n" + "=" * 60)
    print("Test 2: Vague question (should ask for clarification)")
    print("=" * 60)

    result2 = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content="Write a report about AI")]},
    )

    print(f"\nFinal state keys: {list(result2.keys())}")
    print(f"Research brief: {result2.get('research_brief', 'N/A')}")
    print(f"\nMessages ({len(result2['messages'])}):")
    for msg in result2["messages"]:
        print(f"  [{msg.__class__.__name__}] {msg.content[:150]}...")


asyncio.run(main())
