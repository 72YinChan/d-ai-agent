from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from embeddings import get_embeddings


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


persist_directory = Path("./chroma_db_war_and_peace_bge_small_en_v1.5")
model_name_str = "BAAI/bge-small-en-v1.5"

if not persist_directory.exists():
    print(f"错误: 知识库文件 {persist_directory} 未找到。")
    print("请先运行's03_build_index.py'生成向量数据库，再运行该文件")
    exit()

# --- 模块A：链接本地 Chroma 向量数据库 ---
# 1. 加载 Embedding 模型
print(f"正在加载/下载模型{model_name_str}...")
embedding_model = get_embeddings(model_name=model_name_str, device="cpu")
# 2. 加载 Chroma DB
db = Chroma(persist_directory=str(persist_directory), embedding_function=embedding_model)
print("--- Chroma 数据库已加载 ---\n")


# --- 模块B (R-A-G Flow) ---
# 1. R - 检索 -- 强化版
# 1.1 基础检索器 (Base Retriever) - 粗召回
base_retriever = db.as_retriever(search_kwargs={"k": 50})  # k 调大到 50
# 1.2 Reranker (重排器) - 精排序 -- 首次运行需要耗时下载
print("正在加载 Reranker 模型 (bge-reranker-base)...")
encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")  # 加载 Reranker 模型
reranker = CrossEncoderReranker(model=encoder, top_n=6)  # 对检索结果进行精排
# 1.3 创建管道封装器
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,  # 用 Chroma 做海选
    base_compressor=reranker,  # 用 Reranker 做精选
)
retriever = compression_retriever
print("--- 检索器已升级为 Reranker 模式 ---\n")

# 2. A - 增强
sys_prompt = """
你是一个博学的历史学家和文学评论家。
请根据以下上下文回答问题。如果上下文**暗示**了答案，即使未明说，也可推理回答。
如果完全无关，请回答“对不起，根据所提供的上下文我不知道”。

[上下文]: {context}
[问题]: {question}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{question}"),
])

# 3. G - 生成
llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

# 4. 辅助函数
format_docs = lambda docs: "\n".join(doc.page_content for doc in docs)

# 5. 组装 RAG 链（LCEL）
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 运行 RAG 链
print("--- 正在运行 RAG 链 ---")
question = "皮埃尔是共济会成员吗？他在其中扮演什么角色？"
response = rag_chain.invoke(question)
print(f"提问: {question}")
print(f"回答: {response}")
