"""
官方库实现示例

使用官方langchain_mcp_adapters库实现的完整MCP应用示例，展示了如何快速集成MCP服务。

✅ 掌握点：
- 官方MultiServerMCPClient的使用方法
- MCP服务的配置与初始化
- LangGraph工作流的构建
- 官方库与自定义组件的结合使用
功能演示：
- 初始化多服务器MCP客户端
- 加载高德地图MCP服务
- 构建基于LangGraph的地理位置助手
- 使用自定义流式输出组件展示结果

💡 这是一个独立的示例应用，展示了如何使用官方库快速实现MCP功能，适合作为实际项目的参考。
"""
import json
import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv("./.env")
MODEL = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")


# === 配置 MCP 服务器 ===
MCP_SERVERS = {
    # # 方式1.1: 云端代理 —— stdio模式
    # "高德地图": {
    #     "transport": "stdio",
    #     "command": "npx",
    #     "args": ["-y", "@amap/amap-maps-mcp-server"],
    #     "env": {**os.environ, "AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY}
    # },

    # # 方式1.2: 云端MCP服务 —— Streamable HTTP模式
    # "高德地图" :{
    #     "transport":"streamable_http",
    #     "url": f"https://mcp.amap.com/mcp?key={AMAP_MAPS_API_KEY}"
    # },

    # # 方式2.1: 本地工具 —— stdio模式
    # "本地天气":{
    #     "transport": "stdio",
    #     "command": "python",
    #     "args": ["-m", "m10_mcp_basics.s01_stdio_server"],
    #     "env": None
    # },

    # 方式2.2:本地MCP服务 —— Streamable HTTP 模式
    # 注:此方法需要提前运行m10的 s02_streamable_http_server.py
    "本地天气":{
        "transport":"streamable_http",
        "url": "http://127.0.0.1:8001/mcp"
    }
}


def build_graph(available_tools):
    """构建图逻辑 (保持不变)"""
    if not available_tools:
        print("⚠️ 未加载任何工具")

    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, base_url=BASE_URL, streaming=True)

    llm_with_tools = llm.bind_tools(available_tools) if available_tools else llm

    # sys_prompt = "你是一个地理位置助手，请根据用户需求调用工具查询信息。"
    sys_prompt = "你是一个天气助手，请根据用户需求调用工具查询信息。"

    async def agent_node(state: MessagesState):
        # 格式化消息，确保ToolMessage的content是字符串
        formatted_messages = []
        for msg in state["messages"]:
            if isinstance(msg,ToolMessage) and not isinstance(msg.content,str):
                # 将list/dict转为JSON字符串
                formatted_messages.append(
                    ToolMessage(
                        content=json.dumps(msg.content,ensure_ascii=False),
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                formatted_messages.append(msg)
        messages = [SystemMessage(content=sys_prompt)] + formatted_messages
        return {"messages": [await llm_with_tools.ainvoke(messages)]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)

    if available_tools:
        workflow.add_node("tools", ToolNode(available_tools))

        def should_continue(state):
            last_msg = state["messages"][-1]
            return "tools" if last_msg.tool_calls else END

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
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


async def main():
    print("🔌 正在初始化 MCP 客户端...")

    client = MultiServerMCPClient(MCP_SERVERS)

    # 显式建立连接并获取工具
    # 注意：这个 client 对象会保持连接，直到脚本结束
    tools = await client.get_tools()
    print(f"✅ 成功加载工具: {[t.name for t in tools]}")

    # 构建并运行
    app = build_graph(tools)
    # query = "帮我查一下杭州西湖附近的酒店"
    query = "帮我查一下今天佛山天气怎么样？"
    # query = "你好"
    await run_agent_with_streaming(app, query)


if __name__ == "__main__":
    asyncio.run(main())
