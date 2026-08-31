"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
import os
from datetime import datetime
from typing import Annotated, Any, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from tavily import AsyncTavilyClient
import httpx
from ddgs import DDGS

from my_deep_research.configuration import Configuration, SearchAPI, KBMode
from my_deep_research.prompts import summarize_webpage_prompt
from my_deep_research.state import Summary

from my_deep_research.knowledge_base import search_knowledge_base, LOCAL_KB_DESCRIPTION
import chromadb

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_mcp_adapters.client import MultiServerMCPClient
# chromadb客户端对象
_chromadb_client = None

# embedding function对象
_ef = None


##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs."""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value


def get_api_key_for_model(model_name: str, config: RunnableConfig = None):
    """Get API key for a specific model from environment variables."""
    model_name = model_name.lower()
    if model_name.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    elif model_name.startswith("google"):
        return os.getenv("GOOGLE_API_KEY")
    elif model_name.startswith("deepseek"):
        return os.getenv("DEEPSEEK_API_KEY")
    return None


def get_tavily_api_key(config: RunnableConfig = None):
    """Get Tavily API key from environment variables."""
    return os.getenv("TAVILY_API_KEY")


##########################
# Tavily Search Tool Utils
##########################

async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.

    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access

    Returns:
        List of search result dictionaries from Tavily API
    """
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))

    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]

    search_results = await asyncio.gather(*search_tasks)
    return search_results


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content using AI model with timeout protection.

    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized

    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    try:
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )

        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0
        )

        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content


TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )

    # Step 2: Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # Step 3: Set up the summarization model
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = configurable.max_content_length

    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        model_provider="deepseek",
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    # Step 4: Create summarization tasks (skip empty content)
    async def noop():
        """No-op function for results without raw content."""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # Step 5: Execute all summarization tasks in parallel
    summaries = await asyncio.gather(*summarization_tasks)

    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


##########################
# DuckDuckGo + Jina Search Tool Utils
##########################

async def jina_fetch_content(url: str, max_length: int = 50000) -> str:
    """Fetch clean webpage content via Jina Reader API.

    Args:
        url: The webpage URL to extract content from
        max_length: Maximum character length to return

    Returns:
        Clean markdown content from the webpage, or empty string on failure
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "text/markdown"}
            )
            if response.status_code == 200:
                return response.text[:max_length]
    except Exception as e:
        logging.warning(f"Jina fetch failed for {url}: {e}")
    return ""


async def duckduckgo_search_async(search_queries: List[str], max_results: int = 5):
    """Execute multiple DuckDuckGo search queries.

    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query

    Returns:
        List of search result dictionaries with title, url, content
    """
    all_results = []
    for query in search_queries:
        try:
            with DDGS(timeout=20) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                all_results.append({"query": query, "results": results})
        except Exception as e:
            logging.warning(f"DuckDuckGo search failed for '{query}': {e}")
            all_results.append({"query": query, "results": []})
        await asyncio.sleep(0.3)
    return all_results


DUCKDUCKGO_SEARCH_DESCRIPTION = (
    "A free web search engine. Use this to search for information about any topic. "
    "Useful for when you need to answer questions about current events or gather research data."
)


@tool(description=DUCKDUCKGO_SEARCH_DESCRIPTION)
async def duckduckgo_search_tool(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results using DuckDuckGo + Jina Reader.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        config: Runtime configuration for model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute DuckDuckGo searches
    search_results = await duckduckgo_search_async(queries, max_results=max_results)

    # Step 2: Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response["results"]:
            url = result.get("href", "")
            if url and url not in unique_results:
                unique_results[url] = {
                    "title": result.get("title", ""),
                    "content": result.get("body", ""),
                    "url": url,
                    "query": response["query"],
                }

    # Step 3: Format output using DuckDuckGo's built-in snippets (no Jina/LLM needed)
    if not unique_results:
        return "No valid search results found. Please try different search queries."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output

@tool(description=LOCAL_KB_DESCRIPTION)
async def local_knowledge_search(queries: List[str]) -> str:
    return await search_knowledge_base(queries)

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


##########################
# Tool Management Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.

    Args:
        search_api: The search API provider to use

    Returns:
        List of configured search tool objects
    """
    if search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search",
        }
        return [search_tool]
    elif search_api == SearchAPI.DUCKDUCKGO:
        search_tool = duckduckgo_search_tool
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search",
        }
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    return []

async def get_knowledge_base_tools(config: RunnableConfig):
    """Get knowledge base tools based on configuration.
    
    Returns:
        List of knowledge base tools
    """
    configurable = Configuration.from_runnable_config(config)
    kb_mode = KBMode(get_config_value(configurable.kb_mode))
    if kb_mode == KBMode.DIRECT:
        return [local_knowledge_search]
    elif kb_mode == KBMode.MCP:
        mcp_kb_url = configurable.mcp_kb_url
        client = MultiServerMCPClient({
            "knowledge-base": {"transport": "http", "url": mcp_kb_url}
        })
        try:
            tools = await client.get_tools()  # → list[BaseTool]
            return tools
        except Exception as e:
            raise RuntimeError(
                f"MCP knowledge base at {mcp_kb_url} is unreachable. "
                f"Start it with: python -m my_deep_research.mcp_server "
                f"or set kb_mode=direct to use the in-process implementation."
            ) from e
    return []

async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including search and reflection tools.

    Args:
        config: Runtime configuration specifying search API settings

    Returns:
        List of all configured and available tools for research operations
    """
    tools = [think_tool]

    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    knowledge_search_tools = await get_knowledge_base_tools(config)
    tools.extend(knowledge_search_tools)
    return tools


async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


MODEL_TOKEN_LIMITS = {
    "deepseek-chat": 64000,
}


def get_model_token_limit(model: str) -> int | None:
    """Get the maximum token limit for a given model."""
    return MODEL_TOKEN_LIMITS.get(model)


def is_token_limit_exceeded(e: Exception, model: str) -> bool:
    """Check if an exception is caused by exceeding the model's token limit."""
    error_str = str(e).lower()
    keywords = [
        "context length",
        "token limit",
        "too many tokens",
        "context_length_exceeded",
        "maximum context",
    ]
    return any(keyword in error_str for keyword in keywords)

def get_chromadb_client():
    """
    Get a ChromaDB client instance.
    
    Returns:
        ChromaDB client
    """
    global _chromadb_client
    if _chromadb_client is None:
        _chromadb_client = chromadb.PersistentClient(path="./chroma_db")
    return _chromadb_client

def get_embedding_function():
    global _ef
    from modelscope import snapshot_download
    from my_deep_research.configuration import Configuration
    if _ef is None:
        model_name = Configuration().embedding_model
        model_dir = snapshot_download(model_name) 
        _ef = SentenceTransformerEmbeddingFunction(model_name=model_dir)
    return _ef