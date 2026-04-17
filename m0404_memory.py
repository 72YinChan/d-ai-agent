"""
记忆功能

添加会话记忆，实现多轮对话。

✅ 掌握点：
- 使用 ChatMessageHistory 存储历史
- RunnableWithMessageHistory 包装 Chain
- 保持上下文连贯性
"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


prompt = ChatPromptTemplate.from_messages([
    ("system", "你非常可爱，说话末尾会带个喵"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

parser = StrOutputParser()

chain = prompt | llm | parser

# 存储所有会话历史(可用数据库替换)
# 此处用字典模拟，也可替换成Redis、SQL等
store = {}


def get_session_history(session_id: str):
    """根据 session_id 获取该用户的聊天历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


runnable_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

session_id = "user_123"
while True:
    user_input = input("\n你: ")
    if user_input == "quit":
        print("拜拜喵!")
        break
    response = runnable_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"AI: {response}")
