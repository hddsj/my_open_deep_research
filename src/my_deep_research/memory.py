from my_deep_research.utils import get_chromadb_client, get_embedding_function
import uuid
from datetime import datetime


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

def save_memory(topic, content):
    """Save a research summary to the memory collection in ChromaDB.
    
    Args:
        topic (str): The research topic for this memory.
        content (str): The compressed research summary to store.
    """
    _memory_collection = _get_memory_collection()
    # 构建元数据并存入一条记忆
    metadatas=[{"topic": topic, "created_at": datetime.now().isoformat()}]
    _memory_collection.add(documents=[content], metadatas=metadatas, ids=[str(uuid.uuid4())])

def retrieve_memory(query, top_k=3):
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
    lines = []

    # 格式化为可注入 prompt 的文本，截取前 200 字避免过长
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        lines.append(f"{i+1}. [{meta['created_at'][:10]}] {meta['topic']}:{doc[:200]}...")
    return "\n".join(lines) if lines else ""
