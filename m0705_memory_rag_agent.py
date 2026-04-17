"""
终极集成 Agent

首次融合 LangChain 六大核心模块，构建带记忆、能自主决策的智能体。

✅ 掌握点：
- 通过根目录 embeddings.py 模块的 get_embeddings() 函数加载向量化模型。
- LLM：ChatOpenAI 作为推理引擎。
- Prompt：含 MessagesPlaceholder("history") 的系统提示。
- Chain：LCEL 构建的 RAG 链 + AgentExecutor。
- Memory：RunnableWithMessageHistory 实现跨轮次上下文记忆。
- Agents：create_tool_calling_agent 驱动 ReAct 循环。
- RAG：以 search_war_and_peace 工具形式注入私有知识能力。
效果：
- Agent 可结合历史（如“他”指皮埃尔）精准调用 RAG，同时支持查天气等外部工具。
依赖前提：
- 必须存在 chroma_db_war_and_peace_bge_small_en_v1.5 目录。
"""
from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory

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


def build_rag_chain(llm_instance):
    print("--- 正在构建 RAG 链 ---")

    persist_directory = Path("./chroma_db_war_and_peace_bge_small_en_v1.5")
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    encoder_model_name = "BAAI/bge-reranker-base"

    if not persist_directory.exists():
        raise FileNotFoundError(f"索引目录{persist_directory}未找到，请先运行 build_index.py")

    # 链接向量数据库
    print(f"正在加载/下载 Embedding 模型: {embedding_model_name}")
    embedding_model = get_embeddings(model_name=embedding_model_name, device="cpu")
    db = Chroma(persist_directory=str(persist_directory), embedding_function=embedding_model)

    # R
    print(f"正在加载 Reranker模型:{encoder_model_name}...")
    base_retriever = db.as_retriever(search_kwargs={"k": 50})
    encoder = HuggingFaceCrossEncoder(model_name=encoder_model_name)
    reranker = CrossEncoderReranker(model=encoder, top_n=6)
    compression_retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=reranker)
    retriever = compression_retriever

    # A
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", "{question}"),
    ])

    format_docs = lambda docs: "\n".join(doc.page_content for doc in docs)

    # R-A-G
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_instance
        | StrOutputParser()
    )
    print("--- RAG 链构建完毕! ---\n")
    return rag_chain


# 2. 将 RAG 链 组装进 Agent 里
def create_agent_with_memory():
    # LLM
    llm = ChatOpenAI(
        model=MODEL,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个强大的助手。你能查天气，也能查《战争与和平》。请尽力回答用户所提的所有问题。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tool
    rag_chain_instance = build_rag_chain(llm_instance=llm)

    @tool
    def search_war_and_peace(query):
        """查询《战争与和平》小说中的内容，包括人物、情节、历史事件等"""
        print(f"\n正在检索《战争与和平》: {query}")
        return rag_chain_instance.invoke(query)

    @tool
    def get_weather(location):
        """获取天气信息"""
        return f"{location}当前天气：23℃，晴，风力2级"

    tools = [get_weather, search_war_and_peace]

    # 创建 Agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 封装 Memory
    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 添加记忆功能
    agent_with_memory = RunnableWithMessageHistory(
        runnable=agent_executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return agent_with_memory


if __name__ == "__main__":
    session_id = "user123"
    agent = create_agent_with_memory()
    while True:
        user_input = input("\n你: ")
        if user_input == "quit":
            print("拜拜~")
            exit()
        response = agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"AI: {response['output']}")
