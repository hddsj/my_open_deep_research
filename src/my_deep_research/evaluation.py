import math
import asyncio
from langchain.chat_models import init_chat_model
from my_deep_research.configuration import Configuration
from my_deep_research.utils import get_api_key_for_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

async def evaluate_retrieval(query, documents, config) -> dict:
    tasks = [_llm_relevance_score(query, doc, config) for doc in documents]
    scores = await asyncio.gather(*tasks)
    metrics = _compute_metrics(list(scores))
    return metrics

def _compute_metrics(scores) -> dict:
    # 1. MRR — 把结果存到 mrr 变量
    mrr = 0
    for i, score in enumerate(scores):
        if score > 0:
            mrr = 1 / (i + 1)
            break  # 找到第一个就停
    
    # 2. Precision@K
    precision_at_k = sum(1 for score in scores if score > 0)
    precision_K = precision_at_k / len(scores) if len(scores) > 0 else 0
    
    # 3. NDCG
    dcg = 0
    for i, score in enumerate(scores):
        dcg += score / math.log2(i + 2)

    # IDCG (理想情况下的DCG，排序后求和)
    ideal_scores = sorted(scores, reverse=True)
    idcg = 0
    for i, score in enumerate(ideal_scores):
        idcg += score / math.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return {
        "mrr": mrr,
        "precision_at_k": precision_K,
        "ndcg": ndcg
    }

async def _llm_relevance_score(query: str, document: str, config) -> int:
    # 模型初始化
    configurable = Configuration.from_runnable_config(config)
    model = init_chat_model(
        model=configurable.compression_model,
        max_tokens=configurable.compression_model_max_tokens,
        api_key=get_api_key_for_model(configurable.compression_model, config),
    )
    # 定义prompt
    prompt = (
        f"请评估以下文档与查询的相关性，只输出一个数字：\n"
        "- 0：不相关，文档内容与查询无关\n"
        "- 1：部分相关，文档涉及查询主题但不直接回答\n"
        "- 2：高度相关，文档直接回答或详细阐述了查询内容\n\n"
        f"查询：{query}\n\n"
        f"文档：{document[:500]}\n\n"
        "只输出 0、1 或 2，不要解释。"
    )
    try:
        response = await model.ainvoke(prompt)
        score = int(response.content.strip())
        logger.info(f"[llm_relevance_score] 查询: '{query[:30]}' → 相关性: {score}")
        return score
    except Exception as e:
        logger.warning(f"[llm_relevance_score] 评分失败: {e}, 默认返回 0")
        return 0