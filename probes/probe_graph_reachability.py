"""get_graph() 是否靠可达性遍历收集边？没有返回标注的 Command 节点会不会截断链条？

骨架：START -> a -> b -> c，其中
    a  用 Command(goto="b") 路由
    b -> c 是显式静态边
    c -> END 是显式静态边

case A: a 没有返回类型标注   ← 对应 generate_outline 的现状
case B: a 有 Command[Literal["b"]] 标注

如果假设成立：case A 里 b->c 和 c->END 都不会出现在边集里（遍历到 a 就断了），
case B 里链条完整。
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


class S(TypedDict):
    x: int


def node_b(s):
    return {"x": 1}


def node_c(s):
    return {"x": 2}


def build(annotated):
    if annotated:
        def a(s) -> Command[Literal["b"]]:
            return Command(goto="b")
    else:
        def a(s):
            return Command(goto="b")

    b = StateGraph(S)
    b.add_node("a", a)
    b.add_node("b", node_b)
    b.add_node("c", node_c)
    b.add_edge(START, "a")
    b.add_edge("b", "c")       # 显式静态边
    b.add_edge("c", END)       # 显式静态边
    return b.compile()


for label, annotated in [("case A  a 无返回标注", False), ("case B  a 有 Command[Literal['b']]", True)]:
    g = build(annotated).get_graph()
    edges = [(e.source, e.target, e.conditional) for e in g.edges]
    print(f"\n--- {label} ---")
    print(f"  nodes: {list(g.nodes)}")
    for s, t, cond in edges:
        print(f"  {s} -> {t}   conditional={cond}")
    print(f"  b->c 在边集里: {any(s == 'b' and t == 'c' for s, t, _ in edges)}")
