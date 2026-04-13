from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from embeddings import get_embeddings


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

SYS_PROMPT = """\
你是一个博学的历史学家和文学评论家。
请根据以下上下文回答问题。如果上下文**暗示**了答案，即使未明说，也可推理回答。
如果完全无关，请回答“对不起，根据所提供的上下文我不知道”。

[上下文]: {context}
[问题]: {question}
"""


# 全局 LLM (供 Agent 和 RAG 共用)
llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

# (1) 构建一个可复用的 RAG 链 (P1 + P2)
def build_rag_chain(llm_instance):
    print("--- 正在构建 RAG 链... ---\n")

    persist_directory = Path("./chroma_db_war_and_peace_bge_small_en_v1.5")
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    encoder_model_name = "BAAI/bge-reranker-base"

    if not persist_directory.exists():
        raise FileNotFoundError(f"索引目录{persist_directory}未找到，请先运行 build_index.py")

    print(f"正在加载/下载 Embedding模型：{embedding_model_name}")
    embedding_model = get_embeddings(model_name=embedding_model_name, device="cpu")
    db = Chroma(persist_directory=str(persist_directory), embedding_function=embedding_model)

    # 1. R - 检索 -- 强化版
    base_retriever = db.as_retriever(search_kwargs={"k": 50})
    print(f"正在加载 Reranker模型:{encoder_model_name}...")
    encoder = HuggingFaceCrossEncoder(model_name=encoder_model_name)
    reranker = CrossEncoderReranker(model=encoder, top_n=6)
    compression_retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=reranker)
    retriever = compression_retriever

    # 2. A - 增强
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", "{question}"),
    ])

    # 3. G - 生成 (llm 已在全局生成)

    # 4. 辅助函数
    format_docs = lambda docs: "\n".join(doc.page_content for doc in docs)

    # 5. 组装 RAG 链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_instance
        | StrOutputParser()
    )
    print("--- RAG 链构建完毕! ---\n")
    return rag_chain


# 初始化 RAG 链
rag_chain_instance = build_rag_chain(llm)


# (2) 封装为标准 Langchain Tool
@tool
def search_war_and_peace(query):
    """查询《战争与和平》小说中的内容，包括人物、情节、历史事件等"""
    print(f"\n正在检索《战争与和平》: {query}")
    return rag_chain_instance.invoke(query)


# 也可以与其他工具并列使用
@tool
def get_weather(location):
    """获取天气信息"""
    return f"{location}当前天气：23℃，晴，风力2级"


tools = [search_war_and_peace, get_weather]


# 运行
if __name__ == "__main__":
    question = "皮埃尔是共济会成员吗？他在其中扮演什么角色？"
    res = search_war_and_peace.invoke(question)
    print(f"问题: {question}")
    print(f"回答: {res}")
