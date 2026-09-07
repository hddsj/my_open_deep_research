"""生成检索评测查询集的草稿，供人工修订后使用。

做法：随机抽取 child chunk，让 LLM 为每段反推一个「这段能回答的问题」。
chunk 自带 source/page metadata，因此 (问题, 页码) 天然成对，无需人工翻书。

已知偏差（必须在 README 中交代）：
    LLM 反推问题时倾向复用原文措辞，生成的查询偏「字面匹配型」，
    会系统性高估 BM25、低估向量检索。prompt 中已要求改写措辞，但无法消除。
    因此本脚本的输出是**草稿**，必须人工逐条审阅、改写或剔除。

用法：
    python eval/make_queryset.py                 # 生成 eval/queryset_draft.json
"""

import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# load_dotenv() 目前只在 web/server.py 里调用，脚本入口拿不到 .env。
# 长期应提到包级入口，此处先局部加载。
load_dotenv()

import my_deep_research.knowledge_base as kb
from my_deep_research.configuration import Configuration
from my_deep_research.utils import get_api_key_for_model

PER_BOOK = 5              # 每本书抽几段
LEN_RANGE = (300, 800)    # 避开退化 chunk（最短 4 字、最长 19532 字）
SEED = 20260904           # 固定随机种子，保证可复现
OUT = Path(__file__).parent / "queryset_draft.json"

# 明确不在知识库范围内的查询，用于测「库里没有答案时的行为」
OUT_OF_SCOPE = [
    "如何用 React 实现一个下拉菜单组件",
    "咖啡豆的烘焙程度怎么区分",
    "房贷等额本息和等额本金有什么区别",
]

PROMPT = """下面是一段技术书籍的内容。请提出一个真实用户可能会问的问题，该问题能由这段内容回答。

要求：
1. 用**完全不同的措辞**提问，尽量不要复用原文里的词汇和句式
2. 像真人提问那样自然，不要写成「根据上文……」这类考试题
3. 问题要具体，不要宽泛到整本书都能回答
4. 只输出问题本身，不要解释、不要引号

内容：
{chunk}"""


def sample_chunks():
    """按书分组抽样，返回 [(source, page, chunk_text), ...]。"""
    by_book = {}
    for text, meta in zip(kb._bm25_chunks, kb._bm25_metadatas):
        if meta["type"] != "child":
            continue
        if not (LEN_RANGE[0] <= len(text) <= LEN_RANGE[1]):
            continue
        by_book.setdefault(meta["source"], []).append((meta["source"], meta["page"], text))

    rng = random.Random(SEED)
    picked = []
    for source in sorted(by_book):
        pool = by_book[source]
        picked += rng.sample(pool, min(PER_BOOK, len(pool)))
        print(f"  {source}: 候选 {len(pool)} 段，抽 {min(PER_BOOK, len(pool))} 段")
    return picked


async def gen_query(model, chunk):
    resp = await model.ainvoke(PROMPT.format(chunk=chunk))
    return resp.content.strip().strip('"').strip("“”")


async def main():
    folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
    if kb._bm25_index is None:
        kb.build_index(kb.load_documents(folder), folder)

    print("抽样：")
    picked = sample_chunks()

    config = Configuration()
    model = init_chat_model(
        model=config.compression_model,
        max_tokens=config.compression_model_max_tokens,
        api_key=get_api_key_for_model(config.compression_model),
    )

    print(f"\n生成 {len(picked)} 个问题...")
    queries = await asyncio.gather(*(gen_query(model, c[2]) for c in picked))

    items = [
        {
            "id": f"q{i:02d}",
            "query": q,
            "expected_source": src,
            "expected_pages": [page],
            "in_scope": True,
            "reviewed": False,                 # ← 人工审阅后改为 true
            "chunk_excerpt": text[:150],       # 供人工核对，不参与评测
        }
        for i, (q, (src, page, text)) in enumerate(zip(queries, picked), start=1)
    ]
    items += [
        {
            "id": f"x{i:02d}",
            "query": q,
            "expected_source": None,
            "expected_pages": [],
            "in_scope": False,
            "reviewed": True,
            "chunk_excerpt": None,
        }
        for i, q in enumerate(OUT_OF_SCOPE, start=1)
    ]

    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}（{len(items)} 条，其中 {len(OUT_OF_SCOPE)} 条为库外查询）")
    print("下一步：逐条审阅，把明显在抄原文的问题改写或删除，并把 reviewed 改为 true。")


if __name__ == "__main__":
    asyncio.run(main())
