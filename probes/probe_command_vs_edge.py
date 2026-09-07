"""Command(goto=...) 与静态边并存时，目标节点到底跑几次。

三个 case 用同一个骨架，只改「goto 指向哪」和「有没有静态边」：
    case1  goto=END  + 静态边 a->z     ← 对应 need_clarification=True 那条路径
    case2  goto="z"  + 静态边 a->z     ← 对应 allow_clarification=False 那条路径
    case3  goto="z"  无静态边           ← 对照组，验证计数本身是对的

log 用 operator.add 累加（照 state.py:98-101 的写法），否则两次 update 会互相覆盖，
看不出节点跑了几次。
"""

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


class S(TypedDict):
    log: Annotated[list[str], operator.add]


def build(goto, static_edge):
    b = StateGraph(S)
    b.add_node("a", lambda s: Command(goto=goto, update={"log": ["a"]}))
    b.add_node("z", lambda s: {"log": ["z"]})
    b.add_edge(START, "a")
    if static_edge:
        b.add_edge("a", "z")
    return b.compile()


CASES = [
    ("case1  goto=END  + 静态边 a->z ", END, True),
    ("case2  goto='z'  + 静态边 a->z ", "z", True),
    ("case3  goto='z'  无静态边(对照) ", "z", False),
]

for label, goto, static_edge in CASES:
    result = build(goto, static_edge).invoke({"log": []})
    log = result["log"]
    print(f"{label}  log={log}   z 执行 {log.count('z')} 次")
