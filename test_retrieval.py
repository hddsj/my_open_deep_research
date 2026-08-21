"""检索效果评估脚本 - 不影响源代码"""

from my_deep_research.knowledge_base import load_documents, build_index, search

# 测试查询：涵盖 factoid、analytical、multi-hop 三种类型
TEST_QUERIES = [
    # Factoid - 精确事实查询
    {"query": "Docker网络模式", "type": "factoid", "expected_source": "Docker从入门到实践"},
    {"query": "Dockerfile", "type": "factoid", "expected_source": "Docker从入门到实践"},
    {"query": "Python列表的切片操作", "type": "factoid", "expected_source": "Python编程"},
    {"query": "Django Web框架", "type": "factoid", "expected_source": "Python编程"},
    # Analytical - 分析型查询
    {"query": "容器编排和部署策略", "type": "analytical", "expected_source": "Docker从入门到实践"},
    {"query": "Python函数的参数传递方式", "type": "analytical", "expected_source": "Python编程"},
    # Multi-hop - 跨文档查询
    {"query": "Docker容器间通信", "type": "multi-hop", "expected_source": "Docker从入门到实践"},
    {"query": "用Python开发Web应用并部署到Docker", "type": "multi-hop", "expected_source": "Python编程"},
    {"query": "Python单元测试", "type": "factoid", "expected_source": "Python编程"},
    {"query": "Docker数据卷管理", "type": "factoid", "expected_source": "Docker从入门到实践"},
]


def evaluate():
    print("=" * 80)
    print("检索效果评估 (Baseline)")
    print("=" * 80)

    # 先重建索引（使用最新的 chunk_size 和 overlap 参数）
    print("\n正在加载文档并建索引...")
    docs = load_documents("knowledge_base")
    print(f"加载了 {len(docs)} 页")
    build_index(docs)
    print("索引构建完成\n")

    # 逐个测试
    total = len(TEST_QUERIES)
    hits = 0

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        expected = test["expected_source"]
        qtype = test["type"]

        results = search(query, 5)
        docs_found = results["documents"][0]
        metas = results["metadatas"][0]

        # 检查期望来源是否在结果中
        sources = [m["source"] for m in metas]
        hit = any(expected in s for s in sources)
        if hit:
            hits += 1

        # 打印结果
        status = "✅" if hit else "❌"
        print(f"\n--- Query {i+1}/{total} [{qtype}] ---")
        print(f"查询: {query}")
        print(f"期望来源包含: {expected}")
        print(f"结果: {status}")
        for j, (doc, meta) in enumerate(zip(docs_found, metas)):
            print(f"  [{j+1}] {meta['source']} (p{meta['page']}) | {doc[:80]}...")

    # 汇总
    print(f"\n{'=' * 80}")
    print(f"总计: {hits}/{total} 命中 ({hits/total*100:.0f}%)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    evaluate()
