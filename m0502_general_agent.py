import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor


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
    ("system", "你是一个聪明的智能助手。当你遇到解决不了的问题时，会调用工具来解决问题。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # 必须添加，Agent的思考过程
])


@tool
def get_weather(location):
    """模拟获取天气信息"""
    return f"{location}当前天气：23℃，晴，风力2级"


@tool
def get_user_name(user):
    """模拟获取用户名"""
    return f"用户名为：{user}"


tools = [get_weather, get_user_name]

# 创建 Agent
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
# 创建AgentExecutor（执行器） -- 负责运行 ReAct 循环
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  #  开启 verbose 以看到 AI 思考链
# 运行
response = agent_executor.invoke({"input": "今天北京的天气怎么样？"})

print(f"{response}\n{response['output']}")
