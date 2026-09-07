"""图接线的不变量断言。

为什么需要这个文件：LangGraph 里用 `Command(goto=...)` 路由的节点，必须在返回
类型标注 `-> Command[Literal[...]]` 里手工声明所有可能的目标。而框架只校验
「目标是一个已注册的节点名」，不校验「这个节点真的会跳过去」。于是有三种失效：

    1. 静态边与 Command 路由冲突   compile 不报错，运行时该停不停
    2. 漏写标注                    compile 不报错，get_graph() 把节点当死胡同，
                                   并截断下游遍历（孤儿节点 + 显式 add_edge 消失）
    3. 标注写错但是真节点名        compile 不报错，画出一条永远走不到的假边

三种都不会让 compile 或运行时抛异常，也不会影响实际派发 —— 所以没有任何现成机制
会告诉你标注漂移了。这些断言是这条不变量唯一的执行者。

一个实测得来的关键细节（决定了下面为什么分两种数据源）：
    静态边与 Command 边指向同一目标时，`get_graph()` 会把两者去重成一条，
    conditional 仍为 True —— 也就是说 **从 get_graph() 分辨不出静态边是否存在**。
    所以静态边必须查 builder.edges，不能查编译后的图。
    在 get_graph() 上断言「出边恰好两条」抓不到失效 1，那样的断言永远不会红。

首次导入会拉起 torch / sentence-transformers，约十几秒，不是卡死。
不需要 API key，不联网。
"""

from langgraph.graph import END, START

from my_deep_research.deep_researcher import (
    deep_researcher,
    deep_researcher_builder,
)

# 用 add_edge 显式声明的静态边。Command 路由的目标不在这里。
EXPECTED_STATIC_EDGES = {
    (START, "clarify_with_user"),
    ("research_supervisor", "final_report_generation"),
    ("final_report_generation", "evaluate_report"),
}

# 由返回类型标注推导出的条件边。每个 Command(goto=X) 的 X 都应在此出现一条。
EXPECTED_ROUTED_EDGES = {
    ("clarify_with_user", "generate_outline"),
    ("clarify_with_user", END),
    ("generate_outline", "write_research_brief"),
    ("write_research_brief", "research_supervisor"),
    ("evaluate_report", "write_research_brief"),
    ("evaluate_report", END),
}


def _graph_edges():
    """(source, target, conditional) 三元组集合。"""
    return {
        (e.source, e.target, e.conditional)
        for e in deep_researcher.get_graph().edges
    }


def test_static_edges_are_exactly_declared():
    """静态边不能多也不能少。

    对应失效 1：曾经有一条 add_edge("clarify_with_user", "generate_outline")
    与该节点的 Command 路由并存，导致 Command(goto=END) 之后 generate_outline
    仍被执行 —— 澄清环节被绕过，用户拿到的是基于未澄清问题生成的大纲。

    必须查 builder 而不是 get_graph()：见模块 docstring 里的去重说明。
    """
    assert deep_researcher_builder.edges == EXPECTED_STATIC_EDGES, (
        f"静态边集合变了。\n"
        f"  多出来: {deep_researcher_builder.edges - EXPECTED_STATIC_EDGES}\n"
        f"  少掉了: {EXPECTED_STATIC_EDGES - deep_researcher_builder.edges}\n"
        f"如果是有意新增的边，请连同这里的期望值一起更新，并确认它不与某个"
        f"节点的 Command 路由冲突。"
    )


def test_no_orphan_nodes():
    """每个节点都必须至少出现在一条边里。

    对应失效 2：get_graph() 从 START 做可达性遍历，遇到没有返回标注的
    Command 节点就当死胡同接到 END 并停止遍历，导致其下游节点全部悬空 ——
    连显式写的 add_edge 都不会出现在图里。
    """
    edges = _graph_edges()
    connected = {node for src, tgt, _ in edges for node in (src, tgt)}
    orphans = set(deep_researcher.get_graph().nodes) - connected
    assert not orphans, (
        f"以下节点没有任何边: {sorted(orphans)}\n"
        f"通常是某个用 Command 路由的上游节点漏了 "
        f"-> Command[Literal[...]] 标注，导致遍历在那里截断。"
    )


def test_command_routed_edges_are_declared_and_conditional():
    """Command 路由的目标必须全部声明，且被识别为条件边。

    对应失效 2 和 3：标注漏写时这条边会以 conditional=False 的形式接到 END
    （被当成死胡同），标注写错时会出现一条代码里并不存在的目标。
    """
    edges = _graph_edges()
    routed = {(src, tgt) for src, tgt, cond in edges if cond}

    assert routed == EXPECTED_ROUTED_EDGES, (
        f"条件边集合变了。\n"
        f"  多出来: {routed - EXPECTED_ROUTED_EDGES}（代码里可能没有对应的 goto）\n"
        f"  少掉了: {EXPECTED_ROUTED_EDGES - routed}（可能漏了返回类型标注）"
    )

    # 声明过目标的节点，其出边不该再有非条件边混进来
    declared_sources = {src for src, _ in EXPECTED_ROUTED_EDGES}
    stray = {
        (src, tgt)
        for src, tgt, cond in edges
        if src in declared_sources and not cond
    }
    assert not stray, (
        f"这些节点用 Command 路由，却有非条件出边: {sorted(stray)}\n"
        f"要么是漏了标注被当成死胡同，要么是多了一条静态边。"
    )
