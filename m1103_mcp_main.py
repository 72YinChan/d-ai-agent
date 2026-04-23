"""
完整应用示例

使用自定义MCP客户端组件实现的完整应用示例，展示了整个组件系统的协作使用。

✅ 掌握点：
- 组件系统的整体架构
- 多MCP服务的配置与管理
- LangGraph工作流的构建
- 资源的统一管理
功能演示：
- 配置多个MCP服务（云端和本地）
- 批量加载MCP工具
- 构建基于LangGraph的智能体
- 使用流式输出展示结果

💡 这是整个组件系统的完整演示，展示了如何使用自定义实现构建功能完整的MCP应用。

┌─────────────────────────────────────────────────────────┐
│                     应用层                               │
│  ┌───────────────┐  ┌────────────────────────────────┐  │
│  │  mcp_main.py  │  │ final_mcp_main.py (官方库)      │  │
│  └───────────────┘  └────────────────────────────────┘  │
│              │                     │                    │
└──────────────┼─────────────────────┼────────────────────┘
               │                     │
┌──────────────┼─────────────────────┼────────────────────┐
│                     集成层                               │
│  ┌───────────────┐                ┌─────────────────┐   │
│  │  mcp_bridge.py│                │ agent_stream.py │   │
│  └───────────────┘                └─────────────────┘   │
│              │                                          │
└──────────────┼──────────────────────────────────────────┘
               │
┌──────────────┼──────────────────────────────────────────┐
│                     客户端层                              │
│  ┌───────────────┐                                      │
│  │  mcp_client.py│                                      │
│  └───────────────┘                                      │
│              │                                          │
└──────────────┼──────────────────────────────────────────┘
               │
┌──────────────┼──────────────────────────────────────────┐
│                     传输层                               │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │
│  │ transports/   │  │ transports/   │  │ transports/ │  │
│  │   base.py     │  │   http.py     │  │   stdio.py  │  │
│  └───────────────┘  └───────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
"""
import asyncio
from contextlib import AsyncExitStack
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from mcp_bridge import LangChainMCPAdapter


load_dotenv("./.env")
env_vars = os.environ.copy()
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")


MCP_SERVER_CONFIGS = [
    # # 方式1.1: 云端代理 —— stdio 模式
    # {
    #     "name": "高德地图",  # 打印使用了什么 MCP，可移除
    #     "transport": "stdio",  # 指定传输模式
    #     "command": "npx",
    #     "args": ["-y", "@amap/amap-maps-mcp-server"],
    #     "env": env_vars,
    # },

    # # 方式1.2: 云端 MCP 服务 —— Streamable HTTP 模式
    # {
    #     "name": "高德地图",
    #     "transport": "http",
    #     "url": f"https://mcp.amap.com/mcp?key={AMAP_MAPS_API_KEY}"
    # },

    # 方式2.1: 本地工具 —— stdio 模式
    {
        "name": "本地天气",
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "m1001_stdio_server"],
        "env": None,
    },

    # # 方式2.2: 本地 MCP 服务 —— Streamable HTTP 模式
    # {
    #     "name": "本地天气",
    #     "transport": "http",
    #     "url": "http://127.0.0.1:8001/mcp",
    # },
]


SYS_PROMPT = """\
你是一个专业的地理位置服务助手。
1. 当用户查询模糊地点（如"西站"）时，会优先使用相关工具获取具体经纬度或标准名称。
2. 如果用户查询"附近"的店铺，请先确定中心点的坐标或具体位置，再进行搜索。
3. 调用工具时，参数要尽可能精确。
"""


# === 构建图逻辑 ===
def build_graph(available_tools):
    """
    这个函数只认 tools 列表，不关心 tools 的来源
    """
    if not available_tools:
        print("⚠️ 当前没有注入任何工具，Agent 将仅靠 LLM 回答")
    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, base_url=BASE_URL, streaming=True)
    # 如果没工具，bind_tools 会被忽略或处理，LangGraph 同样能正常跑纯对话
    llm_with_tools = llm.bind_tools(available_tools) if available_tools else llm

    # sys_prompt = SYS_PROMPT
    sys_prompt = "你是一个天气助手，请根据用户需求调用工具查询信息。"

    async def agent_node(state: MessagesState):
        messages = [SystemMessage(content=sys_prompt)] + state["messages"]
        # ainvoke: 异步调用版的 invoke
        return {"messages": [await llm_with_tools.ainvoke(messages)]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)

    # 动态逻辑：如果有工具才加工具节点，否则就是纯对话
    if available_tools:
        tool_node = ToolNode(available_tools)
        workflow.add_node("tools", tool_node)

        def should_continue(state: MessagesState):
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tools"
            return END

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

    return workflow.compile()


async def run_agent_with_streaming(app, query:str):
    """
    通用流式运行器，负责将 LangGraph 的运行过程可视化输出到控制台

    :param app: 编译好的 LangGraph 应用 (workflow.compile())
    :param query: 用户输入的问题
    """
    print(f'\n用户: {query}\n')
    print("🤖 AI: ", end="", flush=True)

    # 构造输入消息
    inputs = {"messages": [HumanMessage(content=query)]}

    # 核心:监听v2版本的事件流(相比v1更全面)
    async for event in app.astream_events(inputs, version="v2"):
        kind = event["event"]

        # 1.监听LLM的流式吐字(嘴在动)
        # if kind == "on_chat_model_stream":
        if kind == "on_chain_stream":
            chunk = event["data"]["chunk"]
            if "agent" in chunk:
                agent_chunk = chunk["agent"]["messages"][-1]
                # 过滤掉空的chunk(有时工具调用会产生空内容)
                if agent_chunk.content:
                    print(agent_chunk.content, end="", flush=True)

        # 2.监听工具开始调用(手在动)
        elif kind == "on_tool_start":
            tool_name = event["name"]
            # 不打印内部包装，只打印自定义的工具
            if not tool_name.startswith("_"):
                print(f"\n\n🔨 正在调用工具: {tool_name} ...")

        # 3.监听工具调用结束(拿到结果)
        elif kind == "on_tool_end":
            tool_name = event["name"]
            if not tool_name.startswith("_"):
                print(f"✅ 调用完成，继续思考...\n")
                print("🤖 AI: ", end="", flush=True)

    print("\n\n😊 输出结束!")


# === 主程序 ===
async def main():
    # 使用 ExitStack 统一管理所有资源的关闭
    async with AsyncExitStack() as stack:
        # A. 插件(MCP)注入阶段 -- 允许为空
        dynamic_tools = await LangChainMCPAdapter.load_mcp_tools(stack, MCP_SERVER_CONFIGS)

        # B. 图构建阶段
        app = build_graph(available_tools=dynamic_tools)

        # C. 运行阶段（流式）
        # query = "帮我查一下杭州西湖附近的酒店"
        query = "帮我查一下今天佛山的天气"
        await run_agent_with_streaming(app, query)


if __name__ == "__main__":
    asyncio.run(main())
