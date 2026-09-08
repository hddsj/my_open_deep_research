"""_collect_top_child_indices 的 child 过滤断言。

背景：BM25 对全库（parent + child 混在一起）打分，原来的代码直接切
scores.argsort()[-top_k:][::-1]，没有按 type 过滤。parent 在全库占 19.2%，
且不是均匀分布、会连续聚集，混进 top_k 会导致两个问题：
    1. 真正参与 RRF 融合的 child 候选数量少于 top_k（parent 占了名额却没资格融合）
    2. 向量那一路有 where={"type": "child"} 过滤，BM25 这一路没有，两路口径不一致

_collect_top_child_indices 从 search() 里抽出来，纯函数、不碰 BM25 索引对象、
不碰网络，可以直接构造假的 metadata 列表单测。
"""

from my_deep_research.knowledge_base import _collect_top_child_indices


def test_all_child_collects_exactly_top_k():
    """全部是 child 的理想情况——没有 parent 干扰，应该收集到刚好 top_k 个，
    且不多扫一条。"""
    metadatas = [{"type": "child"} for _ in range(20)]
    scores_index = list(range(20))  # 假装已经按分数从高到低排好序

    child_indices, scanned = _collect_top_child_indices(scores_index, metadatas, top_k=5)

    assert child_indices == [0, 1, 2, 3, 4], "全是 child 时，应该直接取排名前 5"
    assert scanned == 5, "全是 child 时不需要多扫，扫 5 条就该凑够 5 个"


def test_parents_are_skipped_not_counted():
    """parent 混在中间——不能被收集，但要被跳过（计入扫描次数）而不是让整个
    收集提前结束。对应 ranks 1~14 那组真实数据里，排名 11~14 连续 4 个 parent
    的情况：parent 不能截断收集，必须继续往后扫。"""
    # 排好序的下标 0~9，其中 2、5、6、7、8 是 parent（故意让 parent 聚集在一起，
    # 模拟真实数据里 parent 连续出现的情况，而不是均匀散布）
    metadatas = [
        {"type": "child"},   # 0
        {"type": "child"},   # 1
        {"type": "parent"},  # 2
        {"type": "child"},   # 3
        {"type": "child"},   # 4
        {"type": "parent"},  # 5
        {"type": "parent"},  # 6
        {"type": "parent"},  # 7
        {"type": "parent"},  # 8
        {"type": "child"},   # 9  <- 第 5 个 child，在这里才凑够
    ]
    scores_index = list(range(10))

    child_indices, scanned = _collect_top_child_indices(scores_index, metadatas, top_k=5)

    assert child_indices == [0, 1, 3, 4, 9], (
        "应该收集到 0,1,3,4,9 这 5 个 child，中间的 5 个 parent（2,5,6,7,8）都被跳过"
    )
    assert scanned == 10, "凑够 5 个 child 需要扫完全部 10 条（因为 parent 连续聚集在后半段）"


def test_not_enough_children_returns_fewer_without_crashing():
    """候选池里 child 总数不够 top_k——这是正常边界情况，不是异常。

    业界共识（Elasticsearch/Dify 等社区讨论）：请求 top_k 但候选不够时，
    返回不足数量是预期行为，不该报错、也不该用低质量结果强行凑数。
    """
    metadatas = [
        {"type": "child"},
        {"type": "parent"},
        {"type": "child"},
        {"type": "parent"},
    ]
    scores_index = list(range(4))

    child_indices, scanned = _collect_top_child_indices(scores_index, metadatas, top_k=10)

    assert child_indices == [0, 2], "只有 2 个 child，应该老老实实只返回这 2 个"
    assert scanned == 4, "扫完了全部 4 条候选（数组耗尽），不会报错也不会死循环"


def test_scanned_count_can_exceed_top_k():
    """扫描条数应该反映真实付出的代价，即使这个代价比 top_k 大很多。

    对应实测过的真实场景：查询 'ENTRYPOINT 和 CMD 的区别'，top_k=10 时，
    因为排名 11~14 连续 4 个 parent，实际要扫到第 15 条才凑够 10 个 child。
    这里用等价结构复现同一个模式：前 10 个里混 1 个 parent，
    紧接着连续 4 个 parent，第 15 个才是第 10 个 child。
    """
    metadatas = (
        [{"type": "child"}] * 1        # 0        排名1
        + [{"type": "parent"}]         # 1        排名2 parent
        + [{"type": "child"}] * 8      # 2~9      排名3~10
        + [{"type": "parent"}] * 4     # 10~13    排名11~14 连续4个parent
        + [{"type": "child"}] * 1      # 14       排名15 第10个child
    )
    scores_index = list(range(len(metadatas)))

    child_indices, scanned = _collect_top_child_indices(scores_index, metadatas, top_k=10)

    assert len(child_indices) == 10
    assert scanned == 15, "跟实测数据一致：凑够 10 个 child 要扫到第 15 条"
