"""改写重搜时，改写后的查询必须真的被送进搜索工具。

对应 commit 07ff22b 修的 bug：质量评估判定不合格后，重试用的还是原始查询。
这里直接测 _retry_with_rewritten_query，不触碰 LLM 和网络——
execute_tool_safely 被替换成一个记录调用参数的假实现。
"""

import asyncio

from my_deep_research import deep_researcher as dr

REWRITTEN = "改写后的查询"

TOOL_CALLS = [
    {"name": "tavily_search", "args": {"queries": ["原始查询"], "topic": "general"}, "id": "c1"},
    {"name": "think_tool", "args": {"reflection": "原始思考"}, "id": "c2"},
    {"name": "local_knowledge_search", "args": {"queries": ["原始查询"]}, "id": "c3"},
]
OBSERVATIONS = ["旧的网页结果", "原始思考", "旧的知识库结果"]


def _run_retry(monkeypatch):
    """跑一次改写重搜，返回 (新结果列表, 假工具收到的调用记录)。"""
    calls = []

    async def fake_execute_tool_safely(tool, args, config):
        calls.append({"tool": tool, "args": args})
        return f"新结果@{args['queries'][0]}"

    monkeypatch.setattr(dr, "execute_tool_safely", fake_execute_tool_safely)

    tools_by_name = {tc["name"]: f"<tool {tc['name']}>" for tc in TOOL_CALLS}
    observations = asyncio.run(
        dr._retry_with_rewritten_query(
            TOOL_CALLS, OBSERVATIONS, REWRITTEN, tools_by_name, config={}
        )
    )
    return observations, calls


def test_rewritten_query_reaches_search_tools(monkeypatch):
    """原 bug：搜索工具收到的还是原始查询。"""
    _, calls = _run_retry(monkeypatch)

    assert len(calls) == 2, f"应只重跑 2 个搜索工具，实际重跑 {len(calls)} 个"
    for call in calls:
        assert call["args"]["queries"] == [REWRITTEN], (
            f"{call['tool']} 收到的查询不是改写后的: {call['args']['queries']}"
        )


def test_non_search_tools_are_untouched(monkeypatch):
    """think_tool 不该被重跑，它的结果必须原样保留。"""
    observations, calls = _run_retry(monkeypatch)

    assert observations[1] == OBSERVATIONS[1], (
        f"think_tool 的结果被改动了: {observations[1]!r}"
    )
    assert all("think" not in str(c["tool"]) for c in calls), (
        f"think_tool 被错误地重跑了: {calls}"
    )


def test_other_args_are_preserved(monkeypatch):
    """只覆盖 queries，args 里其他字段必须保留。"""
    _, calls = _run_retry(monkeypatch)

    tavily_args = calls[0]["args"]
    assert tavily_args.get("topic") == "general", (
        f"queries 之外的字段丢失了: {tavily_args}"
    )


def test_result_order_matches_tool_calls(monkeypatch):
    """替换后的结果必须与 tool_calls 一一对应，顺序不能错位。"""
    observations, _ = _run_retry(monkeypatch)

    assert len(observations) == len(TOOL_CALLS)
    assert observations[0] == f"新结果@{REWRITTEN}"
    assert observations[2] == f"新结果@{REWRITTEN}"
