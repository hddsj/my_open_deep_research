import os
import jieba
from my_deep_research.knowledge_base import load_documents, build_index, _get_knowledge_client_collection

import my_deep_research.knowledge_base as kb
if kb._bm25_index is None:
    folder_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
    documents = load_documents(folder_path)
    build_index(documents, folder_path)

query = "Docker 网络模式"
top_k = 10

client, collection, ef = _get_knowledge_client_collection()        
results = collection.query(
    query_texts=[query],
    n_results=top_k,
    where={"type": "child"}
)
print(results["distances"][0])
tokenized_query = list(jieba.cut(query))
# 每个文本块的得分
scores = kb._bm25_index.get_scores(tokenized_query)  
# 取最高的 k 个索引
top_indices = scores.argsort()[-top_k:][::-1]