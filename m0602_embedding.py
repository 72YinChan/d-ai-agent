"""
文本向量化

演示如何加载本地 Embedding 模型，并将任意文本（知识卡片）转换为"语义向量"。

✅ 掌握点：
- 使用 HuggingFaceEmbeddings 加载开源本地模型（如 BAAI/bge-small-zh-v1.5）。
- 理解 Embedding（向量化）是 RAG 的"语义标签"，用于实现"语义搜索"。
- 如何调用 embed_query 将文本转换为向量（一串数字）。
"""
from embeddings import get_embeddings


# 首次运行可能时间较久 -- 同时运行本文件需要梯子，不然无法加载到本地
print("--- 正在加载本地 Embedding 模型(BAAI/bge-small-zh-v1.5)... ---")

# 理论：有 embedding 的向量模型
embeddings_model = get_embeddings("BAAI/bge-small-zh-v1.5")
print("Embedding 模型加载完毕")

# 演示：将文本转换为向量
text = "模块05的目标是什么"
query_embedding = embeddings_model.embed_query(text)  # 将文本转换为向量

# 验证：向量存在，而且有具体数值
print(f"文本: {text}")
print(f"向量（前5维）: {query_embedding[:5]}")
print(f"向量维度: {len(query_embedding)}")
