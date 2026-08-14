import os
import pymupdf
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from modelscope import snapshot_download
import jieba
import pickle
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
_bm25_index = None
_bm25_chunks = []
_bm25_metadatas = []

ENCODING_MODEL= "BAAI/bge-small-zh-v1.5"

_reranker = None

_model_dir = None

# chromadb客户端对象
_chromadb_client = None
# embedding_function对象
_ef = None
# knowledge_collection对象
_knowledge_collection = None


def _get_reranker():
    """
    获取reranker对象
    """
    global _reranker
    if _reranker is None:
        model_dir = snapshot_download("BAAI/bge-reranker-base", ignore_file_pattern=["onnx/*"])
        _reranker = CrossEncoder(model_dir)
    return _reranker

def _get_knowledge_client_collection():
    """
    获取知识库客户端、集合和embedding_function对象
    """
    global _knowledge_collection,_chromadb_client,_ef
    if _chromadb_client is None:
        _chromadb_client = chromadb.PersistentClient(path="./chroma_db")
    if _ef is None:
        _ef = SentenceTransformerEmbeddingFunction(model_name=_get_model_dir())
    if _knowledge_collection is None:
        _knowledge_collection = _chromadb_client.get_or_create_collection("knowledge_base", embedding_function=_ef)
    
    return _chromadb_client,_knowledge_collection, _ef

def _get_model_dir():
    """
    获取模型目录
    """
    global _model_dir
    if _model_dir is None:
        _model_dir = snapshot_download(ENCODING_MODEL)
    return _model_dir

def load_documents(folder_path):
    """
    Load all PDF documents from a folder and extract text from each page.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
        
    Returns:
        list: List of dictionaries containing text, source file, and page number
    """
    # 列出当前文件夹下的所有文件
    files = os.listdir(folder_path)  
    documents = []
    # 遍历当前文件夹
    for file in files:
        # 查找当前文件夹里所有pdf文件
        if file.endswith(".pdf"):
            # 打开pdf文件
            doc = pymupdf.open(os.path.join(folder_path,file))
            # 遍历pdf文件的每一页
            for page in doc:
                text = page.get_text()
                # 将每一页的内容添加到documents列表中
                documents.append({"text": text, "source": file, "page": page.number})
    return documents

# 全局语义分块器，避免重复加载 embedding 模型
_semantic_chunker = None

def _get_semantic_chunker():
    global _semantic_chunker
    if _semantic_chunker is None:
        model_dir = _get_model_dir()
        embeddings = HuggingFaceEmbeddings(model_name=model_dir)
        _semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return _semantic_chunker

def split_text(text):
    """
    Split text into semantic chunks based on meaning boundaries.
    
    Args:
        text (str): Text to split
        
    Returns:
        list: List of text chunks
    """
    if not text or not text.strip():
        return []
    chunker = _get_semantic_chunker()
    docs = chunker.create_documents([text])
    return [doc.page_content for doc in docs]
    

def build_index(documents,folder_path):
    """
    Build a knowledge base index from documents.
    
    Args:
        documents (list): List of dictionaries containing text, source file, and page number
        folder_path (str): Path to the folder containing PDF files
        
    Returns:
        chromadb.Collection: ChromaDB collection with the documents indexed
    """
    # 全局变量,用于存储BM25索引
    global _bm25_index, _bm25_chunks, _bm25_metadatas
    # 全局变量，用于存储chromadb客户端、集合和embedding函数
    global _knowledge_collection,_chromadb_client,_ef
    # 获取签名缓存
    cache_path = "./bm25_cache.pkl"
    fingerprint = get_fingerprint(folder_path)

    if os.path.exists(cache_path):
        cache = pickle.load(open(cache_path, "rb"))
        if cache["fingerprint"] == fingerprint:
            _bm25_index = cache["bm25_index"]
            _bm25_chunks = cache["bm25_chunks"]
            _bm25_metadatas = cache["bm25_metadatas"]
            return
        
    all_ids = []
    
    client, collection, ef = _get_knowledge_client_collection()        

    books = {}
    # 按书名分组
    for doc in documents:
        source = doc["source"]
        if source not in books:
            books[source] = ""
        # 拼接文本（插入页码标记）
        books[source] += f"[PAGE:{doc['page']}]" + doc["text"]

    # 遍历每本书
    for source, text in books.items():   
        chunks = split_text(text)
        # 判断是否为空白页,若是空白页,则不加入collection
        if not chunks:
            continue
        for i, chunk in enumerate(chunks):
            # 通过正则表达式提取页码
            pages = re.findall(r'\[PAGE:(\d+)\]', chunk)
            page = int(pages[0]) if pages else 0
            # 去掉页码标记
            clean_chunk = re.sub(r'\[PAGE:\d+\]', '', chunk)
            # 收集数据
            all_ids.append(f"{source}_{page}_{i}")
            _bm25_metadatas.append({"source": source, "page": page})
            _bm25_chunks.append(clean_chunk)
    # 建立chromdb知识库索引
    batch_size = 5000
    for i in range(0, len(_bm25_chunks), batch_size):
        collection.add(
            documents=_bm25_chunks[i:i+batch_size],
            metadatas=_bm25_metadatas[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )
    # 分词
    tokenized = [list(jieba.cut(chunk)) for chunk in _bm25_chunks]
    # 根据分词结果建立BM25索引
    _bm25_index = BM25Okapi(tokenized)
    pickle.dump({
        "fingerprint": fingerprint,
        "bm25_index": _bm25_index,
        "bm25_chunks": _bm25_chunks,
        "bm25_metadatas": _bm25_metadatas,
    }, open(cache_path, "wb"))
    return collection

def search(query, top_k):
    """
    Search the knowledge base for relevant documents.
    
    Args:
        query (str): Search query
        top_k (int): Number of results to return
        
    Returns:
        dict: Search results with documents and metadata
    """
    client, collection, ef = _get_knowledge_client_collection()        
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    tokenized_query = list(jieba.cut(query))
    # 每个文本块的得分
    scores = _bm25_index.get_scores(tokenized_query)  
    # 取最高的 k 个索引
    top_indices = scores.argsort()[-top_k:][::-1]
    
    # RRF 合并 (Reciprocal Rank Fusion)
    k = 60
    merged = {}

    # 向量结果 RRF 分数
    for rank, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        key = doc[:100]
        if key not in merged:
            merged[key] = {"document": doc, "metadata": meta, "score": 0}
        merged[key]["score"] += 1 / (k + rank + 1)

    # BM25 结果 RRF 分数
    for rank, idx in enumerate(top_indices):
        doc = _bm25_chunks[idx]
        meta = _bm25_metadatas[idx]
        key = doc[:100]
        if key not in merged:
            merged[key] = {"document": doc, "metadata": meta, "score": 0}
        merged[key]["score"] += 1 / (k + rank + 1)

    # 按 RRF 分数降序排列
    merged_list = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    # Cross-Encoder 重排
    reranker = _get_reranker()

    pairs = [[query, item["document"]] for item in merged_list]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, merged_list), key=lambda x: x[0], reverse=True)
    merged_list = [item for _, item in ranked]
    
    return {
        "documents": [[item["document"] for item in merged_list]],
        "metadatas": [[item["metadata"] for item in merged_list]],
    }

def get_fingerprint(folder_path):
    """
    获取文件夹下所有pdf文件的修改时间作为签名
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        dict: 文件名到修改时间的映射
    """
    # 列出文件夹下的所有文件
    files = os.listdir(folder_path)  
    documents = {}
    # 遍历文件夹
    for file in files:
        # 查找文件夹里所有pdf文件
        if file.endswith(".pdf"):
            # 获取文件的修改时间作为签名
            full_path = os.path.join(folder_path, file)
            mtime = os.path.getmtime(full_path)
            documents[file] = mtime
    
    return documents
            