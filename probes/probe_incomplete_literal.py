"""标注列得不全时，跳到没列的目标会怎样？

case A: 标注只声明 "b"，运行时 goto "c"   ← 不全的标注
case B: 完全没有标注，运行时 goto "c"      ← 对照，已知可用

关心两件事：
    1. compile() 过不过
    2. invoke() 时真的跳到 c 了吗，还是报错 / 静默走别的路
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


class S(TypedDict):
    trail: list


def build(annotated):
    if annotated:
        # 只声明 b，实际却 goto c
        def a(s) -> Command[Literal["b"]]:
            return Command(goto="c", update={"trail": s["trail"] + ["a"]})
    else:
        def a(s):
            return Command(goto="c", update={"trail": s["trail"] + ["a"]})

    b = StateGraph(S)
    b.add_node("a", a)
    b.add_node("b", lambda s: {"trail": s["trail"] + ["b"]})
    b.add_node("c", lambda s: {"trail": s["trail"] + ["c"]})
    b.add_edge(START, "a")
    b.add_edge("b", END)
    b.add_edge("c", END)
    return b.compile()


for label, annotated in [
    ('case A  标注只声明 "b"，实际 goto "c"', True),
    ('case B  无标注，实际 goto "c"', False),
]:
    print(f"\n--- {label} ---")
    try:
        app = build(annotated)
        print("  compile: OK")
    except Exception as e:
        print(f"  compile: {type(e).__name__}: {e}")
        continue
    try:
        out = app.invoke({"trail": []})
        print(f"  invoke : OK   trail={out['trail']}")
    except Exception as e:
        print(f"  invoke : {type(e).__name__}: {e}")
    edges = [(e.source, e.target, e.conditional) for e in app.get_graph().edges]
    print(f"  图里的边: {edges}")
