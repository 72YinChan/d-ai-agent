"""
人工审批机制

实现带有人工干预的智能体工作流，为敏感操作添加审批环节。

✅ 掌握点：
- 在 LangGraph 中使用 interrupt_before=["tools"] 设置中断点
- 通过 app.get_state(config) 获取工作流当前状态
- 实现用户交互审批流程，控制高危操作的执行
- 结合 MemorySaver 实现持久化状态管理
输出：
- 当智能体尝试执行敏感操作（如发送邮件）时，系统会暂停并请求用户确认
- 用户输入 "yes" 批准执行，输入其他取消操作
- 完整展示执行过程的日志和最终结果

💡 此示例适用于需要人工监督的场景，如金融交易、系统配置修改等高风险操作。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


# LLM 配置
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, base_url=BASE_URL)

# 定义一个敏感工具：发送邮件（模拟）
@tool
def send_email(to, content):
    """发送邮件"""
    return f"邮件已发送至 {to}，内容为：{content}"


# 工具绑定到 LLM
tools = [send_email]
llm_with_tools = llm.bind_tools(tools)

# Node 函数与 Edge 节点
tool_node = ToolNode(tools)


def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# 构建基础 ReAct 图
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile(
    checkpointer=MemorySaver(),  # 在内存里做状态持久化
    interrupt_before=["tools"],  # 选择要人工审批的节点 -- 负责在哪里停，之后的代码负责停了之后怎么办
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user123"}}
    user_input = "请帮我给 boss@example.com 发一封邮件，内容是：会议推迟到明天下午3点。不要询问其他细节。"
    print("用户输入:", user_input)
    print("\nAgent 正在思考...\n")

    # 初识输入
    inputs = {"messages": [HumanMessage(content=user_input)]}

    while True:
        # 触发工作流执行，推进到下一个中断点或自然结束
        # inputs 注入事件
        # config 确定回话 id
        # "values": 完整记录每步结果
        for _ in app.stream(inputs, config, stream_mode="values"):  # 流式执行
            pass  # 必须迭代生成器，才能实际执行工作流

        # 获取当前状态
        snapshot = app.get_state(config)
        next_tasks = snapshot.next  # 返回下一步要执行的节点名列表

        # 如果没有下一步，说明工作流已结束
        if not next_tasks:
            final_msg = snapshot.values["messages"][-1]
            print(f"\n最终回复: {final_msg.content}")
            break

        # 如果下一步是需要审批的节点
        if "tools" in next_tasks:
            last_msg = snapshot.values["messages"][-1]
            tool_call = last_msg.tool_calls[0]
            print(f'\n⚠️ Agent准备执行操作：')
            print(f'    工具名称：{tool_call["name"]}')
            print(f'    参数：{tool_call["args"]}')

            approval = input("\n✅ 是否批准执行？(输入 'yes' 继续，其他取消): ").strip().lower()
            if approval == "yes":
                print("\n继续执行...")
                inputs = None  # 表示从断点继续，无新输入
            else:
                print("\n❌ 操作已取消，流程终止")
                break
