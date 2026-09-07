"""对比两路召回的分数量纲，并观察 parent 对 BM25 召回名额的挤占。

背景：
- 向量路在 search() 里有 where={"type": "child"} 过滤，只在 child 中检索
- BM25 路没有该过滤，parent 与 child 一起参与打分，直到 search() 第 377 行才被跳过
  → 两路的召回口径不一致

三类查询用于观察两路各自的强弱，以及库中没有答案时的表现。
"""

import os

import jieba

import my_deep_research.knowledge_base as kb

TOP_K = 10
PARENT_RATIO_IN_CORPUS = 0.192  # 1528 / 7941

QUERIES = [
    ("关键词型", "ENTRYPOINT 和 CMD 的区别"),
    ("概念型", "容器之间是怎么互相通信的"),
    ("库外型", "如何用 React 写一个下拉菜单"),
]


def ensure_index():
    """确保 BM25 索引与 ChromaDB 就绪（build_index 内含一致性校验）。"""
    if kb._bm25_index is None:
        folder = os.path.join(os.path.dirname(__file__), "knowledge_base")
        kb.build_index(kb.load_documents(folder), folder)


def probe(query, top_k=TOP_K):
    _, collection, _ = kb._get_knowledge_client_collection()

    # 向量路：与 search() 一致，只检索 child
    vec = collection.query(query_texts=[query], n_results=top_k, where={"type": "child"})
    distances = vec["distances"][0]

    # BM25 路：对全库打分，parent 与 child 混在一起
    scores = kb._bm25_index.get_scores(list(jieba.cut(query)))
    ranked = scores.argsort()[::-1]

    # (a) 未过滤的 top_k —— 用于量化 parent 占了多少名额
    raw_top = ranked[:top_k]
    n_parent = sum(1 for i in raw_top if kb._bm25_metadatas[i]["type"] == "parent")

    # (b) 只取 child，凑满 top_k，并记录为此扫描了多少条
    child_top, scanned = [], 0
    for i in ranked:
        scanned += 1
        if kb._bm25_metadatas[i]["type"] == "child":
            child_top.append(i)
            if len(child_top) == top_k:
                break

    bm25_max, bm25_min = scores[child_top[0]], scores[child_top[-1]]
    vec_min, vec_max = min(distances), max(distances)

    print(f"\n--- {query} ---")
    print(f"  向量 distance    {vec_min:.4f} ~ {vec_max:.4f}    (越小越相似)")
    print(f"  BM25  score      {bm25_min:.3f} ~ {bm25_max:.3f}    (越大越相关)")
    print(f"  量级比           BM25max / distmax = {bm25_max / vec_max:.0f}x")
    print(f"  未过滤 top{top_k} 中 parent 占 {n_parent}/{top_k}"
          f"  (全库基准 {PARENT_RATIO_IN_CORPUS:.1%})")
    print(f"  凑满 {top_k} 个 child 需扫描 {scanned} 条")


if __name__ == "__main__":
    ensure_index()
    for kind, q in QUERIES:
        print(f"\n===== {kind} =====")
        probe(q)
