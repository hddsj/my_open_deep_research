from my_deep_research.utils import get_chromadb_client, get_embedding_function, get_api_key_for_model
import uuid
import logging
from datetime import datetime
from my_deep_research.configuration import Configuration
from langchain.chat_models import init_chat_model
from collections import defaultdict
logger = logging.getLogger(__name__)

# 研究记忆集合对象（模块级缓存）
_memory_collection = None

def _get_memory_collection():
    """Get or create the research_memory collection in ChromaDB.
    
    Uses module-level caching to avoid repeated collection creation.
    
    Returns:
        chromadb.Collection: The research_memory collection instance.
    """
    global _memory_collection
    if _memory_collection is None:
        # 获取 ChromaDB 客户端和 embedding 函数
        _chromadb_client = get_chromadb_client()
        _ef = get_embedding_function()
        # 获取或创建 research_memory 集合
        _memory_collection = _chromadb_client.\
                                get_or_create_collection("research_memory", embedding_function=_ef)
    return _memory_collection

def _merge_memories(old_content, new_content):
    try:
        config = Configuration()
        model = init_chat_model(
                model=config.compression_model,
                max_tokens=config.compression_model_max_tokens,
                api_key=get_api_key_for_model(config.compression_model),
            )
        prompt = (
                "请将以下两段研究摘要合并为一段完整的摘要。"
                "保留两者的所有关键信息，去除重复内容，保持简洁。\n\n"
                f"【已有摘要】：\n{old_content}\n\n"
                f"【新研究】：\n{new_content}"
            )
        response = model.invoke(prompt)
        logger.info(f"[_merge_memories] 合并成功，合并后摘要前100字: {str(response.content)[:100]}...")
        return response.content
    except Exception as e:
        logger.error(f"[_merge_memories] LLM合并失败，降级为新内容: {e}")
        return new_content

def _reflect_on_memories():
    """Reflect on existing memories to check for duplicates or updates.
    
    Returns:
        dict: The query results from ChromaDB.
    """
    # 重新获取集合实例
    _memory_collection = _get_memory_collection()
    
    if _memory_collection.count() > 0:
       
        try:
            # 只取 type=research 的普通记忆，返回格式：
            # results = {
            #     "documents": ["Docker bridge用veth...", "K8s通过PV/PVC...", ...],  ← 一维列表
            #     "metadatas": [
            #         {"type": "research", "topic": "Docker网络基础", "created_at": "2026-08-19T..."},
            #         {"type": "research", "topic": "K8s存储", "created_at": "2026-08-19T..."},
            #         ...
            #     ],
            #     "ids": ["uuid1", "uuid2", ...]
            # }
            # 注意：get() 返回一维列表，不像 query() 返回嵌套列表
            results = _memory_collection.get(where={"type": "research"})
            logger.info(f"[_reflect_on_memories] 取出 {len(results['documents'])} 条普通记忆")
            # 将results["documents"]和results["metadatas"]按topic分组
            groups = defaultdict(list)
            for doc, meta in zip(results["documents"], results["metadatas"]):
                groups[meta["topic"]].append(doc)
            # 模型初始化
            config = Configuration()
            model = init_chat_model(
                    model=config.compression_model,
                    max_tokens=config.compression_model_max_tokens,
                    api_key=get_api_key_for_model(config.compression_model),
                )
            
            logger.info(f"[_reflect_on_memories] 按topic分为 {len(groups)} 组: {list(groups.keys())}")
            for topic, docs in groups.items():
                docs_text = "\n".join(f"- {d}" for d in docs)
                prompt = f"请总结以下关于'{topic}'的研究记录，保留关键信息，不超过150字：\n\n{docs_text}"
                response = model.invoke(prompt)
                # 存入集合，type=topic_summary
                _memory_collection.add(documents=[response.content], metadatas=[{"type": "topic_summary","topic": topic, "created_at": datetime.now().isoformat()}], ids=[str(uuid.uuid4())])
                logger.info(f"[_reflect_on_memories] 主题摘要已生成: [{topic}] {str(response.content)[:80]}...")
            
            topic_summaries = _memory_collection.get(where={"type": "topic_summary"})
            
            all_summaries = "\n".join(f"- {doc}" for doc in topic_summaries["documents"])
            
            # 第二步：基于主题摘要生成领域洞察
            reflection_prompt = (
                "请基于以下各主题的研究摘要，进行跨主题的元认知反思：\n"
                "1. 用户关注哪些核心领域？各领域的研究深度如何？\n"
                "2. 研究方向有什么趋势或演变？\n"
                "3. 各主题之间有什么关联或互补？\n"
                "请输出一段简洁的领域洞察（不超过300字）。\n\n"
                f"【各主题摘要】：\n{all_summaries}"
            )
            reflection_response = model.invoke(reflection_prompt)
            metadatas=[{"type": "reflection","topic": "反思总结", "created_at": datetime.now().isoformat()}]
            _memory_collection.add(documents=[reflection_response.content], metadatas=metadatas, ids=[str(uuid.uuid4())])
            logger.info(f"[_reflect_on_memories] 领域洞察已生成: {str(reflection_response.content)[:100]}...")
            return reflection_response.content
        except Exception as e:
            logger.error(f"[_reflect_on_memories] LLM反思总结失败: {e}")
            return None

def save_memory(topic, content):
    """Save a research summary to the memory collection in ChromaDB.
    
    Args:
        topic (str): The research topic for this memory.
        content (str): The compressed research summary to store.
    """
    _memory_collection = _get_memory_collection()
    # 去重：检查是否已存在高度相似的记忆
    if _memory_collection.count() > 0:
        existing = _memory_collection.query(query_texts=[content], n_results=1)
        # 判断是否存在高度相似的记忆
        if existing["distances"][0] and existing["distances"][0][0] < 0.1:
            logger.info(f"[save_memory] 检测到相似记忆 (distance={existing['distances'][0][0]:.4f})，执行合并")
            # 获取旧记忆
            old_content = existing["documents"][0][0]
            # 合并记忆
            content = _merge_memories(old_content, content)
            # 删除旧记忆，后续 add 会存入更新的版本
            _memory_collection.delete(ids=existing["ids"][0])
        else:
            logger.info(f"[save_memory] 无相似记忆，直接存入新记忆")
    # 构建元数据并存入一条记忆
    metadatas=[{"type": "research", "topic": topic, "created_at": datetime.now().isoformat()}]
    _memory_collection.add(documents=[content], metadatas=metadatas, ids=[str(uuid.uuid4())])
    # 每5条记忆进行一次反思总结
    if _memory_collection.count() % 5 == 0:
        try:
            _reflect_on_memories()
        except Exception as e:
            logger.error(f"[_reflect_on_memories] 反思总结失败: {e}")

def retrieve_memory(query, top_k=3,max_distance=1.0):
    """Retrieve relevant past research memories from ChromaDB via vector similarity search.
    
    Args:
        query (str): The search query to find related past research.
        top_k (int): Maximum number of memories to retrieve.
    
    Returns:
        str: Formatted string of related memories, or empty string if none found.
    """
    _memory_collection = _get_memory_collection()

    # 空集合直接返回，避免查询报错
    if _memory_collection.count() == 0:
        return ""
    
    # 向量相似度检索相关记忆
    results = _memory_collection.query(query_texts=[query], n_results=top_k)

    # 解析结果：ChromaDB query 返回嵌套列表，结构如下：
    # {
    #     "documents": [["摘要1", "摘要2", ...]],  ← 外层列表对应每个 query
    #     "metadatas": [[{"topic": "...", "created_at": "..."}, ...]],
    #     "distances": [[0.3, 0.5, ...]]
    # }
    # 因为只传了一个 query，所以取 [0] 获取第一个 query 的结果
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    lines = []

    # 格式化为可注入 prompt 的文本，截取前 200 字避免过长
    for i, (doc, meta,dist) in enumerate(zip(docs, metas,distances)):
        # 如果相似度大于max_distance，说明相关性低，则跳过
        if dist > max_distance:
            continue
        lines.append(f"{i+1}. [{meta['created_at'][:10]}] {meta['topic']}:{doc[:200]}...")
    return "\n".join(lines) if lines else ""
