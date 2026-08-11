import os
import pymupdf
import chromadb

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

def split_text(text, chunk_size=500, overlap=100):
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
    # 获取chromadb客户端(类似于数据库的客户端)
    client = chromadb.PersistentClient(path="./chroma_db")
    # 获取或创建知识库集合(类似于数据库的表)
    collection = client.get_or_create_collection("knowledge_base")
    for doc in documents:
        chunks = split_text(doc["text"])
        # 判断是否为空白页,若是空白页,则不加入collection
        if not chunks:
            continue
        # 添加文档到知识库集合中
        collection.add(
            documents=chunks,
            metadatas=[{"source": doc["source"], "page": doc["page"]} for _ in chunks],
            ids=[f"{doc['source']}_{doc['page']}_{i}" for i in range(len(chunks))]
        )
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
    return results