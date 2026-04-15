"""
多智能体协作系统

构建由总控节点协调多个专家智能体的协作系统，实现专业化分工。

✅掌握点：
- 定义共享的多智能体状态结构（AgentState）
- 创建不同功能的专家节点（RAG专家、网络研究员、代码编写者）
- 实现 Supervisor 总控节点，负责任务分配和流程协调
- 为每个专家配置专属工具和系统指令
- 设计复杂的条件路由逻辑，支持多节点协作
效果：
- 系统会根据用户问题自动选择最合适的专家处理
- 专家可以调用专业工具获取信息
- 总控节点监控对话进度，决定是否完成任务或切换专家
- 完整展示多智能体交互过程和决策路径

💡 此架构适用于需要多领域专业知识协作解决的复杂问题，如技术咨询、客户服务等场景。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


SYS_PROMPT = """
你是一个任务协调员。你的目标是管理专家来解决用户的问题。

当前对话需要以下专家参与：
- rag_expert：涉及公司政策、内部流程
- web_research：涉及外部新闻、公开数据
- code_writer：需要生成代码

【决策逻辑】
1. **检查历史记录**：先看上一个回复是否已经完整回答了用户的初始问题。
2. **如果已经回答完毕**：必须输出 'FINISH'。
3. **如果尚未回答或需要补充**：根据当前缺少的步骤，选择下一个最合适的专家。

请只输出专家名字或 'FINISH'，不要输出任何其他解释。
"""


# llm配置
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, base_url=BASE_URL)

# 模拟工具
@tool
def search_internal_docs(query:str):
    """搜索公司内部文档获取政策信息"""
    return "根据公司手册，年假为15天"


@tool
def search_web(query:str):
    """通过搜索引擎获取最新的公开信息"""
    return "据TechCrunch报道，LangGraph 0.6已支持持久化记忆"


@tool
def generate_code(requirement:str):
    """根据需求生成可运行的Python代码"""
    return "python\nprint('Hello from Code writer!')"


# 共享状态定义
class AgentState(MessagesState):
    next_speaker: str


# 专家节点
def rag_expert(state:AgentState):
    prompt = "你是公司知识库专家，只基于内部文档回答问题。回答应简洁明了，直接给出最终结论。请在回答的最后一行加上：(任务已完成)"
    messages = [SystemMessage(content=prompt)]+state['messages']
    tools = [search_internal_docs]
    response = llm.bind_tools(tools).invoke(messages)
    return {'messages':[response]}


def web_research(state:AgentState):
    prompt = "你是互联网研究员，擅长用搜索引擎获取最新公开信息。回答应简洁明了，直接给出最终结论。请在回答的最后一行加上：(任务已完成)"
    messages = [SystemMessage(content=prompt)]+state['messages']
    tools = [search_web]
    response = llm.bind_tools(tools).invoke(messages)
    return {'messages':[response]}


def code_writer(state:AgentState):
    prompt = "你是python工程师，只生成可运行代码，不解释。回答应简洁明了，直接给出最终结论。请在回答的最后一行加上：(任务已完成)"
    messages = [SystemMessage(content=prompt)]+state['messages']
    tools = [generate_code]
    response = llm.bind_tools(tools).invoke(messages)
    return {'messages':[response]}


# 总控节点
def supervisor(state:AgentState):
    supervisor_prompt = SYS_PROMPT
    messages = [SystemMessage(content=supervisor_prompt)]+state['messages']
    response = llm.invoke(messages)
    next_speaker = response.content.strip()
    return {"next_speaker":next_speaker}


# 路由函数定义
def route_supervisor(state:AgentState):
    if state["next_speaker"]=="FINISH":
        return END
    return state["next_speaker"]


def should_continue(state:AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg,"tool_calls") and last_msg.tool_calls:
        return "tools"
    return "supervisor"


def route_after_tool(state:AgentState):
    # 工具执行完后，通过next_speaker知道是谁调用的，路由回去
    return state["next_speaker"]


# 添加工具节点
tools = [search_internal_docs, search_web, generate_code]
tool_node = ToolNode(tools)


# 构建协作图
workflow = StateGraph(AgentState)
# 1. 添加节点
workflow.add_node("supervisor", supervisor)
workflow.add_node("rag_expert", rag_expert)
workflow.add_node("web_research", web_research)
workflow.add_node("code_writer", code_writer)
workflow.add_node("tools", tool_node)
# 2. 总控回路
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", route_supervisor)
# 3. 专家节点的ReAct循环
# 为每个专家添加条件边：决定是去执行工具还是回总控
for member in ["rag_expert", "web_research", "code_writer"]:
    workflow.add_conditional_edges(
        member,
        should_continue,
        {"tools": "tools", "supervisor": "supervisor"}
    )
# 4.工具节点闭环
# 工具执行完，根据next_speaker路由回原来的专家
workflow.add_conditional_edges("tools", route_after_tool)

app = workflow.compile()
# 保存可视化架构图
with open('wf_multi_agent_orchestration.png', 'wb') as f:
    f.write(app.get_graph().draw_mermaid_png())
print("图表已保存为 wf_multi_agent_orchestration.png")


# 测试运行
if __name__ == '__main__':
    user_input = "公司年假多少天"
    print("用户提问:", user_input)
    print('\n开始多智能体协作...\n')

    inputs = {"messages": [HumanMessage(content=user_input)]}
    # app是编译好的图，stream()会让图开始运转，并返回一个生成器
    # 图每执行完一个节点，就会产出一个step字典
    for step in app.stream(inputs):
        # 因为step是个字典，所以需要拆包拿到 节点名(Node) 与 输出内容(output)
        for node, output in step.items():
            # 有工具/专家回复
            if "messages" in output:
                msg = output["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    call = msg.tool_calls[0]
                    print(f"【{node}】调用工具 {call['name']}({call['args']})")
                else:
                    print(f"【{node}】回复：{msg.content}")
            # supervisor刚做完决策，确定下个发言人
            elif "next_speaker" in output:
                speaker = output["next_speaker"]
                print(f"【Supervisor】指定下一位发言人：{speaker}")
