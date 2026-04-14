import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


# LLM 配置
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, base_url=BASE_URL)

# 工具定义
@tool
def get_weather(location):
    """获取天气"""
    return f"{location}当前天气：23℃，晴，风力2级"


tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)  # 让 llm 学会调用工具节点

# --- 核心组件：拆解 AgentExecutor ---
# ReAct Step1: Thought（LLM 决策）
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}  # 新消息追加到状态


# ReAct Step2-3: Action + Observation
tool_node = ToolNode(tools)  # 工具节点函数，langgraph 已封装

# ReAct Step4: Loop Controller（是否循环）
def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"  # 有工具调用 -> 执行工具
    return END  # 无工具调用 -> 返回答案


# --- 构建 ReAct 循环图 ---
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)  # Thought
workflow.add_node("tools", tool_node)  # Action + Observation

workflow.add_edge(START, "agent")

# 条件边: Thought -> 决定下一步
workflow.add_conditional_edges(
    "agent",  # 从哪个节点出发
    should_continue,  # 决定下一步去哪
    {
        "tools": "tools",  # 如果返回 tools，去 tools 节点
        END: END,          # 如果返回 END，直接结束工作流
    }
)

workflow.add_edge("tools", "agent")  # 工具调用的结果再返回给 agent 节点

app = workflow.compile()
# 保存可视化架构图
with open('wf_react_agent.png', 'wb') as f:
    f.write(app.get_graph().draw_mermaid_png())
print("图表已保存为 wf_react_agent.png")


if __name__ == "__main__":
    # 触发工具
    result = app.invoke({"messages": [HumanMessage(content="北京天气如何？")]})
    print("工具调用结果:", result["messages"][-1].content)
    # 不触发工具
    result = app.invoke({"messages": HumanMessage(content="你好")})
    print("直接回答:", result["messages"][-1].content)
