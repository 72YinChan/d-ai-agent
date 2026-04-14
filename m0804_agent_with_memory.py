import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


# LLM 配置
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, base_url=BASE_URL)

# Prompt 配置
sys_prompt = "你是一个强大的助手，能查天气，也能回答一般问题。请使用中文回答。"

# 工具定义
@tool
def get_weather(location):
    """获取天气"""
    return f"{location}当前天气：23℃，晴，风力2级"


tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)  # 让 LLM 学会调用工具节点

# --- 核心组件：拆解 AgentExecutor ---
# ReAct Step1: Thought（LLM 决策）
def call_model(state: MessagesState):
    # 构造带 system_prompt 的完整消息列表（仅用于本次 LLM 调用）
    message_for_llm = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm_with_tools.invoke(message_for_llm)
    # 此处只会返回新生成的消息，不包含 prompt，防止污染历史
    return {"messages": [response]}  # 新消息追加到状态


# ReAct Step2-3: Action + Observation
tool_node = ToolNode(tools)

# ReAct Step4: Loop Controller（是否循环）
def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"  # 有工具调用 -> 执行工具
    return END          # 无工具调用 -> 返回答案


# --- 构建 ReAct 循环图 ---
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)  # Thought
workflow.add_node("tools", tool_node)   # Action + Observation

workflow.add_edge(START, "agent")
# 条件边：Thought -> 决定下一步
workflow.add_conditional_edges(
    "agent",        # 从哪个节点出发
    should_continue,       # 决定下一步去哪
    {
        "tools": "tools",  # 如果返回 tools，去 tools 节点
        END: END,          # 如果返回 END，直接结束工作流
    }
)
workflow.add_edge("tools", "agent")

# 编译时启用记忆
app = workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    session_id = "user123"
    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = input("\n你: ")
        if user_input.strip().lower() == "quit":
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        ai_msg = result["messages"][-1]
        print(f"AI: {ai_msg.content}")
