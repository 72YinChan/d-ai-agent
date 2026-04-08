import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

prompt = ChatPromptTemplate.from_messages([
    ('system','你是小智，一个帮助他人的智能助手。当你无法解答当前问题时，会调用工具来解决问题。'),
    MessagesPlaceholder(variable_name="history"),  # 对话历史记录占位符
    ('human','{input}'),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # Agent 的思考过程占位符
])


# 配置tool
@tool
def get_weather(location):
    """模拟获取天气信息"""
    return f"{location}当前天气：23℃，晴，风力2级"


tools = [get_weather]
# 配置agent
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
# 配置AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 记忆存储 -- 包装 agent_executor
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
    history_messages_key="history"
)
session_id = "user_123"


if __name__ == '__main__':
    while True:
        user_input = input('\n你:')
        if user_input == 'quit':
            print('拜拜~')
            break
        response = agent_with_memory.invoke(
            {'input': user_input},
            config={'configurable': {'session_id': session_id}}
        )
        print(f"AI:{response['output']}")
