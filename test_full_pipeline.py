"""End-to-end test for the full deep researcher pipeline."""

import asyncio

from dotenv import load_dotenv
load_dotenv()

from my_deep_research.deep_researcher import deep_researcher


async def main():
    """Run the full pipeline: clarify → brief → supervisor → final report."""
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": "What is LangGraph and how does it compare to LangChain?"}]},
        config={
            "configurable": {
                "allow_clarification": False,
                "max_researcher_iterations": 2,
                "max_react_tool_calls": 3,
                "max_concurrent_research_units": 2,
            }
        },
    )

    print("=" * 80)
    print("RESEARCH BRIEF:")
    print("=" * 80)
    print(result.get("research_brief", "N/A"))

    print("\n" + "=" * 80)
    print("NOTES (from supervisor):")
    print("=" * 80)
    notes = result.get("notes", [])
    for i, note in enumerate(notes):
        print(f"\n--- Note {i+1} ---")
        print(note[:500] + "..." if len(note) > 500 else note)

    print("\n" + "=" * 80)
    print("FINAL REPORT:")
    print("=" * 80)
    report = result.get("final_report", "N/A")
    print(report[:3000] + "..." if len(report) > 3000 else report)

    print("\n" + "=" * 80)
    print(f"Report length: {len(result.get('final_report', ''))} chars")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
