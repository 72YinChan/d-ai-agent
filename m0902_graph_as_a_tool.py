"""
子图工具化封装

演示如何将一个完整的 LangGraph 子工作流封装为单个工具（Graph-as-a-Tool 模式）。

✅ 掌握点：
- 定义独立的子图状态结构（如 RetryState）
- 构建包含重试逻辑的子图工作流
- 使用 @tool 装饰器将子图封装为可调用工具
- 在主工作流中像使用普通工具一样调用子图
效果：
- 子图内部实现了失败重试逻辑，对调用者完全透明
- 主工作流可以无缝集成复杂的子流程，保持代码整洁
- 模拟实现了不稳定 API 调用的容错处理

💡 这种模式特别适合封装具有复杂内部逻辑的操作，如网络请求、数据库事务等可能需要重试或错误处理的场景。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.messages import SystemMessage, HumanMessage


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


# LLM 配置
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, base_url=BASE_URL)

# 构建子工作流
# 1. 子任务状态
class RetryState(MessagesState):
    query: str
    attempt: int
    result: str


# 2. 子图逻辑 -- 模拟一个可能失败，需重试的 API 调用
def call_unstable_api(state: RetryState):
    """偶发性的外部服务，偶发失败"""
    attempt = state["attempt"]
    if attempt == 1:
        # 第一次故意失败
        return {"result": "ERROR: 服务暂时不可用", "attempt": attempt + 1}
    else:
        # 第二次成功
        return {"result": f"SUCCESS: 成功处理请求: {state['query']}", "attempt": attempt + 1}


def should_retry(state: RetryState):
    if "ERROR" in state["result"] and state["attempt"] <= 2:  # 出现报错且重试次数小于 2，重连
        return "call_api"
    return END


# 3. 构建子图工作流
retry_workflow = StateGraph(RetryState)
retry_workflow.add_node("call_api", call_unstable_api)

retry_workflow.add_edge(START, "call_api")
retry_workflow.add_conditional_edges(
    "call_api",
    should_retry,
    {"call_api": "call_api", END: END},
)

retry_app = retry_workflow.compile()


# 4. 封装为 tool(Graph-as-a-Tool)
@tool
def create_order(query: str) -> str:
    """创建新订单，自动重试保障成功率"""
    result = retry_app.invoke({"query": query, "attempt": 1, "result": ""})
    return result["result"]


# 5. 主 Graph
tools = [create_order]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# 构建主工作流
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# 运行
if __name__ == "__main__":
    user_input = "请创建一个新订单：购买三本书"
    print("用户输入:", user_input)

    inputs = {"messages": [
        SystemMessage(content="你是一个任务执行助手。当用户提出任何需要处理、操作或执行的请求时，必须调用 create_order 工具来完成，不要自行回答细节"),
        HumanMessage(content=user_input),
    ]}
    result = app.invoke(inputs)

    tool_result = None
    # 在主工作流的消息历史中，查找最近的工具执行结果
    for msg in reversed(result["messages"]):
        if msg.type == "tool":  # 找到 ToolMessage 类型消息
            tool_result = msg.content
            break
    if tool_result:
        print(f"\n✅ 直接获取子图返回值:\n{tool_result}")
    else:
        print("\n❌ 未执行任何工具")

    final_reply = result["messages"][-1]
    print(f"\n最终恢复:\n{final_reply}")
