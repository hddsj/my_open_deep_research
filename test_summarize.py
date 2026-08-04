"""Test script to verify the full tavily_search pipeline including AI summarization."""

import asyncio

from dotenv import load_dotenv

from my_deep_research.utils import tavily_search

load_dotenv()

async def main():
    print("Testing full tavily_search pipeline (search + summarize)...")
    print("This will call DeepSeek to summarize each result.\n")

    # Use the @tool function — invoke it with a dict
    result = await tavily_search.ainvoke(
        {"queries": ["What is LangGraph framework"]},
        config={"configurable": {"max_search_results": 1}}
    )

    print(result)

asyncio.run(main())
