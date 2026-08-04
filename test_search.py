"""Test script to verify Tavily search integration works correctly."""

import asyncio

from dotenv import load_dotenv

from my_deep_research.utils import tavily_search_async, think_tool, get_today_str

# Step 1: Load .env file
load_dotenv()

async def main():
    # Step 2: Test helper function
    print(f"Today: {get_today_str()}")
    print()

    # Step 3: Test Tavily search (async)
    print("Searching Tavily for 'What is LangGraph'...")
    results = await tavily_search_async(
        ["What is LangGraph"],
        max_results=2,
        include_raw_content=False  # False to save API quota
    )

    # Step 4: Print results
    for response in results:
        print(f"\nQuery: {response.get('query', 'N/A')}")
        for i, result in enumerate(response['results']):
            print(f"  [{i+1}] {result['title']}")
            print(f"      URL: {result['url']}")
            print(f"      Snippet: {result['content'][:100]}...")
    print()

    # Step 5: Test think_tool
    result = think_tool.invoke({"reflection": "Search returned 2 results about LangGraph."})
    print(f"think_tool output: {result}")

# Run the async main function
asyncio.run(main())
