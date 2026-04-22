"""
通用流式输出组件

实现通用的LangGraph事件流监听和可视化输出功能，提供友好的用户交互体验。

✅ 掌握点：
- LangGraph v2事件流的监听与处理
- LLM流式吐字的实时渲染
- 工具调用过程的可视化展示
- 异步事件处理的最佳实践
功能：
- 监听LLM的流式输出并实时打印
- 显示工具调用的开始和结束状态
- 过滤内部包装工具，只显示自定义工具
- 优化控制台输出格式，提升用户体验

💡 这是一个独立的工具组件，可以与任何LangGraph应用集成，用于增强用户交互体验。
"""
import asyncio

from langchain_core.messages import HumanMessage


async def run_agent_with_streaming(app, query:str):
    """
    通用流式运行器，负责将 LangGraph 的运行过程可视化输出到控制台

    :param app: 编译好的 LangGraph 应用 (workflow.compile())
    :param query: 用户输入的问题
    """
    print(f'\n用户: {query}\n')
    print("🤖 AI:", end="", flush=True)

    # 构造输入消息
    inputs = {"messages": [HumanMessage(content=query)]}

    # 核心:监听v2版本的事件流(相比v1更全面)
    async for event in app.astream_events(inputs, version="v2"):
        kind = event["event"]

        # 1.监听LLM的流式吐字(嘴在动)
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # 过滤掉空的chunk(有时工具调用会产生空内容)
            if chunk.content:
                print(chunk.content, end="", flush=True)

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
