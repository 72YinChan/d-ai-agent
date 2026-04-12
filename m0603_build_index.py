from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings


# 0. 文件准备
knowledge_base_content = """
### 模块 01 — Agent 入门 & 环境搭建

- **目标**：理解 Agent 概念，完成环境配置与首次调用。
- **内容**：环境依赖｜API Key 配置｜最小可运行 Agent

### 模块 02 — LLM 基础调用

- **目标**：掌握模型调用逻辑，初步构建智能体能力。
- **内容**：LLM了解与调用｜Prompt编写与逻辑构思｜多轮对话记忆｜独立搭建一个智能体

### 模块 03 — Function Calling 与工具调用

- **目标**：实现 LLM 调用外部函数，赋予模型“执行力”。
- **内容**：Function calling原理｜工具函数封装｜API接入实践｜多轮调用流程｜Agent能力扩展

### 模块 04 — LangChain 基础篇

- **目标**：认识Langchain六大模块，学会用Langchain构建智能体。
- **内容**：LLM 调用｜Prompt 设计｜Chain 构建｜Memory 记忆｜实战练习

### 模块 05 — LangChain 进阶篇

- **目标**：掌握Langchain Agents的核心机制，构建能调用工具、持续思考、具备记忆的智能体。
- **内容**：Function Calling｜@tool 工具封装｜ReAct 循环｜Agent 构建｜SQL Agent｜记忆+流式｜开发优化

"""
knowledge_base_path = Path("./knowledge_base.txt")
with open(knowledge_base_path, "w", encoding="utf-8") as f:
    f.write(knowledge_base_content)


# 1. 加载并切分文档（Load & Split）
loader = TextLoader(knowledge_base_path, encoding="utf8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)  # 载入切分器模板
splits = text_splitter.split_documents(docs)  # 实际执行切割
print(f"p1完成，文档已切分成{len(splits)}个片段\n")

# 2. 向量化（Embedding）
embeddings_model = get_embeddings()  # 载入 Embedding 模型
print(f"p2完成，Embedding 模型已准备\n")

# 3. 存储（Store）
# 将分割块与向量化的模型传递给 FAISS，FAISS 会使用它们并完成最终向量数据库的构建
db = FAISS.from_documents(splits, embeddings_model)
db.save_local("faiss_index")
print(f"p3完成，向量数据库（db）已构建")

# 4. 清理文件
if knowledge_base_path.exists():
    knowledge_base_path.unlink()

print("--- 所有阶段已经完成！ ---")
