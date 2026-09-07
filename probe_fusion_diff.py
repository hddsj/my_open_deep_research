"""RRF 与「逐 query min-max 归一化 + 加权」两种融合方式，结果到底差多少。

先测差异，再测优劣：如果两者的 top-5 几乎重合，就没有必要为「谁更好」去建标注集。
本脚本不需要任何标注。

两处需要注意的设计决策（min-max 需要回答、RRF 不需要）：
- 向量 distance 越小越相似，加权前必须先转成相似度
- 只出现在单路召回里的候选，另一路给多少分？这里取 0
"""

import os

import jieba

import my_deep_research.knowledge_base as kb

POOL = 20        # 每路召回的候选数
TOP_K = 5        # 最终对比的截断位置
RRF_K = 60       # 与 search() 中一致
ALPHA = 0.5      # min-max 加权中向量的权重

QUERIES = [
    ("关键词型", "ENTRYPOINT 和 CMD 的区别"),
    ("概念型", "容器之间是怎么互相通信的"),
    ("库外型", "如何用 React 写一个下拉菜单"),
]


def ensure_index():
    if kb._bm25_index is None:
        folder = os.path.join(os.path.dirname(__file__), "knowledge_base")
        kb.build_index(kb.load_documents(folder), folder)


def recall_two_paths(query):
    """两路各召回 POOL 条 child，返回 (文本 -> 排名) 与 (文本 -> 原始分数)。"""
    _, collection, _ = kb._get_knowledge_client_collection()

    vec = collection.query(query_texts=[query], n_results=POOL, where={"type": "child"})
    vec_docs = vec["documents"][0]
    vec_dists = vec["distances"][0]

    scores = kb._bm25_index.get_scores(list(jieba.cut(query)))
    bm_docs, bm_scores = [], []
    for i in scores.argsort()[::-1]:
        if kb._bm25_metadatas[i]["type"] != "child":
            continue
        bm_docs.append(kb._bm25_chunks[i])
        bm_scores.append(scores[i])
        if len(bm_docs) == POOL:
            break

    return (vec_docs, vec_dists), (bm_docs, bm_scores)


def fuse_rrf(vec, bm):
    """只用排名。缺席的一路自然不贡献分数，无需额外决策。"""
    vec_docs, _ = vec
    bm_docs, _ = bm
    merged = {}
    for rank, doc in enumerate(vec_docs):
        merged[doc] = merged.get(doc, 0) + 1 / (RRF_K + rank + 1)
    for rank, doc in enumerate(bm_docs):
        merged[doc] = merged.get(doc, 0) + 1 / (RRF_K + rank + 1)
    return sorted(merged, key=merged.get, reverse=True)


def _minmax(values, higher_is_better):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]


def fuse_minmax(vec, bm):
    """逐 query 归一化后加权。缺席的一路记 0 分。"""
    vec_docs, vec_dists = vec
    bm_docs, bm_scores = bm
    vec_norm = dict(zip(vec_docs, _minmax(vec_dists, higher_is_better=False)))
    bm_norm = dict(zip(bm_docs, _minmax(bm_scores, higher_is_better=True)))

    merged = {}
    for doc in set(vec_docs) | set(bm_docs):
        merged[doc] = ALPHA * vec_norm.get(doc, 0.0) + (1 - ALPHA) * bm_norm.get(doc, 0.0)
    return sorted(merged, key=merged.get, reverse=True)


def compare(query):
    vec, bm = recall_two_paths(query)
    a, b = fuse_rrf(vec, bm), fuse_minmax(vec, bm)

    overlap5 = len(set(a[:5]) & set(b[:5]))
    overlap10 = len(set(a[:10]) & set(b[:10]))
    same_top1 = a[0] == b[0]

    # 共同候选的平均名次位移
    pos_a = {d: i for i, d in enumerate(a)}
    pos_b = {d: i for i, d in enumerate(b)}
    common = set(pos_a) & set(pos_b)
    mean_shift = sum(abs(pos_a[d] - pos_b[d]) for d in common) / len(common)

    print(f"\n--- {query} ---")
    print(f"  候选池并集      {len(set(a))} 条")
    print(f"  top-{TOP_K} 重合     {overlap5}/{TOP_K}")
    print(f"  top-10 重合     {overlap10}/10")
    print(f"  第 1 名相同     {same_top1}")
    print(f"  平均名次位移    {mean_shift:.2f}")


if __name__ == "__main__":
    ensure_index()
    for kind, q in QUERIES:
        print(f"\n===== {kind} =====")
        compare(q)
