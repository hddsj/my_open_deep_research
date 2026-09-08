"""_check_research_limits 的两道停止条件断言。

背景：researcher 每轮的停止条件本来只看"轮次"（按查询复杂度分配的轮次上限，
simple/medium/complex 对应 2/3/4）。这里新增第二道独立防线——"工具调用总数"，
因为一轮里 LLM 可以并行发起多个工具调用，轮次没到但调用总数已经异常时，
轮次那道防线管不住。两道防线是 or 关系，任一触发就停。

_check_research_limits 从 researcher_tools 里抽出来，纯函数、不碰网络和 LLM，
可以直接构造假的 state / tool_calls / configurable 单测。
"""

from my_deep_research.configuration import Configuration
from my_deep_research.deep_researcher import _check_research_limits

PRINCLPLE = {"simple": 2, "medium": 3, "complex": 4}


def test_iterations_alone_triggers_stop():
    """轮次到了，工具调用数没到——单靠轮次防线也能停。"""
    state = {"tool_call_iterations": 4, "total_tool_calls": 5}
    configurable = Configuration(max_react_tool_calls=18)

    exceeded, iterations_exceeded, tool_calls_exceeded, new_total = (
        _check_research_limits(state, tool_calls=[{}], princlple=PRINCLPLE,
                                complex_classify="complex", configurable=configurable)
    )

    assert iterations_exceeded is True, "4 轮已达 complex 的上限 4，轮次防线应该触发"
    assert tool_calls_exceeded is False, "累计只到 6，远低于上限 18，这道防线不该触发"
    assert exceeded is True, "两道防线是 or 关系，一道触发整体就该停"


def test_tool_calls_alone_triggers_stop():
    """轮次没到，但某一轮并行发的工具调用把总数顶穿了上限——这正是加这道防线的原因。"""
    state = {"tool_call_iterations": 1, "total_tool_calls": 10}
    configurable = Configuration(max_react_tool_calls=18)

    exceeded, iterations_exceeded, tool_calls_exceeded, new_total = (
        _check_research_limits(state, tool_calls=[{}] * 10, princlple=PRINCLPLE,
                                complex_classify="complex", configurable=configurable)
    )

    assert new_total == 20, "10（已有）+ 10（本轮） 应该是 20"
    assert iterations_exceeded is False, "才 1 轮，complex 允许到 4 轮，远没到"
    assert tool_calls_exceeded is True, "累计 20 已经超过上限 18，这道防线该触发"
    assert exceeded is True, "工具调用防线单独触发，也应该让整体停下来"


def test_neither_triggers_continue():
    """两道防线都没到——应该继续循环，不能停。"""
    state = {"tool_call_iterations": 1, "total_tool_calls": 2}
    configurable = Configuration(max_react_tool_calls=18)

    exceeded, iterations_exceeded, tool_calls_exceeded, new_total = (
        _check_research_limits(state, tool_calls=[{}], princlple=PRINCLPLE,
                                complex_classify="complex", configurable=configurable)
    )

    assert iterations_exceeded is False
    assert tool_calls_exceeded is False
    assert exceeded is False, "两道防线都没触发，不该被停"


def test_tool_calls_boundary_is_inclusive():
    """>= 而不是 >：累计正好等于上限时，也算超限，不能等到超过才停。

    这是设计决策本身的一部分：如果用 >，累计恰好等于上限的那一轮会被放过，
    上限就变成了「摸到线也没事」，失去了兜底的意义。
    """
    configurable = Configuration(max_react_tool_calls=18)

    at_limit = {"tool_call_iterations": 1, "total_tool_calls": 17}
    _, _, exceeded_at_limit, new_total_at_limit = _check_research_limits(
        at_limit, tool_calls=[{}], princlple=PRINCLPLE,
        complex_classify="complex", configurable=configurable,
    )
    assert new_total_at_limit == 18
    assert exceeded_at_limit is True, "累计正好 18，等于上限，也应该算超限"

    below_limit = {"tool_call_iterations": 1, "total_tool_calls": 16}
    _, _, exceeded_below, new_total_below = _check_research_limits(
        below_limit, tool_calls=[{}], princlple=PRINCLPLE,
        complex_classify="complex", configurable=configurable,
    )
    assert new_total_below == 17
    assert exceeded_below is False, "累计 17，还差 1 才到上限，不该被停"
