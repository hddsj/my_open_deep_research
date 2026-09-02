"""跨模式一致性检查。

direct（进程内）与 mcp（MCP server）两种知识库模式，对大模型而言必须无法区分：
工具名、描述、参数 schema 三者必须完全一致，否则切换模式会改变模型行为。

需要 MCP server 运行中。未运行时跳过并明确打印 SKIPPED，不静默通过。
    启动：python -m my_deep_research.mcp_server
"""

import asyncio

import pytest

from my_deep_research.utils import MCPUnreachableError, get_all_tools

START_SERVER_HINT = "python -m my_deep_research.mcp_server"


def schema_of(tool):
    """取出工具的 JSON Schema。

    direct 模式下 args_schema 是 pydantic 模型类，mcp 模式下是 MCP server
    返回的原始 dict（langchain-mcp-adapters 不重建 pydantic 模型）。
    """
    s = tool.args_schema
    return s if isinstance(s, dict) else s.model_json_schema()


async def kb_tool(mode):
    """取指定模式下唯一的知识库工具。"""
    tools = await get_all_tools({"configurable": {"kb_mode": mode}})
    matches = [t for t in tools if "knowledge" in t.name]
    assert len(matches) == 1, (
        f"{mode} 模式下期望恰好 1 个知识库工具，实际工具为 {[t.name for t in tools]}"
    )
    return matches[0]


async def main():
    """执行一致性检查。MCP 服务未启动时抛 MCPUnreachableError，由调用方决定如何处理。"""
    direct = await kb_tool("direct")
    mcp = await kb_tool("mcp")

    assert mcp.name == direct.name, (
        f"工具名不一致: direct={direct.name!r} mcp={mcp.name!r}"
    )
    assert mcp.description == direct.description, (
        f"描述不一致:\n  direct={direct.description!r}\n  mcp={mcp.description!r}"
    )

    d, m = schema_of(direct), schema_of(mcp)
    assert m.get("properties") == d.get("properties"), (
        f"参数不一致:\n  direct={d.get('properties')}\n  mcp={m.get('properties')}"
    )
    assert m.get("required") == d.get("required"), (
        f"必填字段不一致: direct={d.get('required')} mcp={m.get('required')}"
    )

    print(f"ok: 两种模式的知识库工具一致 (name={direct.name})")


def test_kb_mode_parity():
    """pytest 入口。异步逻辑用 asyncio.run 包一层，避免为单个测试引入异步插件。"""
    try:
        asyncio.run(main())
    except MCPUnreachableError as e:
        pytest.skip(f"MCP server 未运行，跨模式一致性检查未执行。启动后重跑: {START_SERVER_HINT}（{e}）")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except MCPUnreachableError as e:
        print("SKIPPED: MCP server 未运行，跨模式一致性检查未执行")
        print(f"         启动后重跑: {START_SERVER_HINT}")
        print(f"         原因: {e}")
