from my_deep_research.knowledge_base import search_knowledge_base, LOCAL_KB_DESCRIPTION
from mcp.server.fastmcp import FastMCP
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

mcp = FastMCP("Knowledge Base", json_response=True)

@mcp.tool(description=LOCAL_KB_DESCRIPTION)
async def local_knowledge_search(queries: list[str]) -> str:
    return await search_knowledge_base(queries)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
