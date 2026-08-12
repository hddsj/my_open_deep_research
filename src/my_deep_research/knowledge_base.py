import os
import pymupdf
import chromadb
from rank_bm25 import BM25Okapi

_bm25_index = None
_bm25_chunks = []
_bm25_metadatas = []

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

def split_text(text, chunk_size=800, overlap=200):
    """
    Split text into chunks with overlap.
    
    Args:
        text (str): Text to split
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    # 分块后的结果
    chunks = []
    # 对每页的内容分块,其中chunk_size是每块的大小，overlap是块之间的重叠部分
    for i in range(0, len(text), chunk_size-overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
    

def build_index(documents):
    """
    Build a knowledge base index from documents.
    
    Args:
        documents (list): List of dictionaries containing text, source file, and page number
        
    Returns:
        chromadb.Collection: ChromaDB collection with the documents indexed
    """
    # 全局变量,用于存储BM25索引
    global _bm25_index, _bm25_chunks, _bm25_metadatas
    # 获取chromadb客户端(类似于数据库的客户端)
    client = chromadb.PersistentClient(path="./chroma_db")
    # 获取或创建知识库集合(类似于数据库的表)
    collection = client.get_or_create_collection("knowledge_base")
    for doc in documents:
        chunks = split_text(doc["text"])
        # 判断是否为空白页,若是空白页,则不加入collection
        if not chunks:
            continue
        # 建立chromdb知识库索引
        collection.add(
            documents=chunks,
            metadatas=[{"source": doc["source"], "page": doc["page"]} for _ in chunks],
            ids=[f"{doc['source']}_{doc['page']}_{i}" for i in range(len(chunks))]
        )

        # 在循环中存所有chunks和metadatas
        _bm25_metadatas.extend([{"source": doc["source"], "page": doc["page"]} for _ in chunks])
        _bm25_chunks.extend(chunks)
    # BM25的chunks分词
    tokenized = [list(chunk) for chunk in _bm25_chunks]
    # 根据分词结果建立BM25索引
    _bm25_index = BM25Okapi(tokenized)
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
    # 获取chromadb客户端(类似于数据库的客户端)
    client = chromadb.PersistentClient(path="./chroma_db")
    # 获取或创建知识库集合(类似于数据库的表)
    collection = client.get_or_create_collection("knowledge_base")
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    tokenized_query = list(query) 
    # 每个文本块的得分
    scores = _bm25_index.get_scores(tokenized_query)  
    # 取最高的 k 个索引
    top_indices = scores.argsort()[-top_k:][::-1]
    
    # 合并结果：用字典去重
    merged = {}

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
         key = doc[:100]  # 用前100字符作为去重key
         if key not in merged:
             merged[key] = {"document": doc, "metadata": meta}
    
    # 再加 BM25 的结果
    for idx in top_indices:
        doc = _bm25_chunks[idx]
        meta = _bm25_metadatas[idx]
        key = doc[:100]
        if key not in merged:
            merged[key] = {"document": doc, "metadata": meta}
    
    merged_list = list(merged.values())
    
    return {
    "documents": [[item["document"] for item in merged_list]],
    "metadatas": [[item["metadata"] for item in merged_list]],
}   