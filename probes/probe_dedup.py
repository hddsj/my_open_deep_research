"""静态边与 Command(goto) 指向同一目标时，get_graph() 显示成几条边？

这个答案决定「clarify_with_user 出边恰好两条」这个断言能不能抓到
「有人把静态边加回去」这个回归。如果去重成一条且 conditional 仍为 True，
断言就抓不到，必须换维度。

case A: 只有 Command[Literal["b"]] 标注（正确状态）
case B: 标注 + 静态边 a->b 并存（回归状态）
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


class S(TypedDict):
    x: int


def build(static_edge):
    def a(s) -> Command[Literal["b"]]:
        return Command(goto="b")

    b = StateGraph(S)
    b.add_node("a", a)
    b.add_node("b", lambda s: {"x": 1})
    b.add_edge(START, "a")
    if static_edge:
        b.add_edge("a", "b")      # 回归：静态边加回来
    b.add_edge("b", END)
    return b.compile()


for label, static_edge in [("case A  只有标注（正确）", False), ("case B  标注+静态边（回归）", True)]:
    g = build(static_edge).get_graph()
    es = [(e.source, e.target, e.conditional) for e in g.edges]
    a_out = [(t, c) for s, t, c in es if s == "a"]
    print(f"\n--- {label} ---")
    print(f"  全部边: {es}")
    print(f"  a 的出边: {a_out}   共 {len(a_out)} 条")
