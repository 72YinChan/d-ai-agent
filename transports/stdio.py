"""
标准输入输出传输实现

实现基于标准输入输出的MCP传输层，支持与本地MCP服务通信。

✅ 掌握点：
- AsyncExitStack资源管理
- 子进程通信的实现
- MCP协议的低级实现
- 异步上下文管理器的应用
功能：
- 启动本地MCP服务进程
- 建立标准输入输出管道通信
- 管理MCP会话生命周期
- 自动清理资源

💡 此实现支持本地MCP服务调用，适合开发和调试阶段使用。
"""
from typing import Optional, List, Dict
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client


class StdioMCPTransport:
    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict] = None,
    ):
        """
        MCP 客户端 - 支持 stdio 与 HTTP 两种传输方式

        :param command: stdio 模式的命令（如 npx）
        :param args: stdio 模式的参数
        :param env: stdio 模式的环境变量
        """
        # MCP 启动方式（npx/uvx/python -m xxx）
        self.params = StdioServerParameters(command=command, args=args, env=env)
        # 工程核心：资源栈
        self.exit_stack = AsyncExitStack()
        # 连接会话（长连接）
        self.session: Optional[ClientSession] = None

    async def connect(self):
        """建立 MCP 长连接（一次连接，多次调用）"""
        if self.session:
            return  # 已连接无需重复
        # 进入 transport（读/写管道）
        transport = await self.exit_stack.enter_async_context(
            stdio_client(self.params)
        )
        # 创建 JSON_RPC 对话
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(transport[0], transport[1])
        )
        # 等待 MCP 服务器返回工具清单
        await self.session.initialize()

    async def list_tools(self):
        """查询工具列表，为 LLM 建立上下文用"""
        if not self.session:
            raise RuntimeError("未连接，请先 connect()")

        result = await self.session.list_tools()

        # 🔍 调试：打印工具的完整信息，确认工具是否被正确封装
        # if result.tools:
        #     import json
        #     # 使用 model_dump() (Pydantic v2) 或 dict() (v1) 查看原始数据
        #     first_tool = result.tools[0]
        #     print(f"\n🔍 [DEBUG] 原始工具数据: {first_tool}\n")

        # 转为纯字典，LLM 能读
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, args: dict):
        """调用工具（工程化：加上防御性处理）"""
        if not self.session:
            raise RuntimeError("未连接，请先 connect()")

        result = await self.session.call_tool(name, args)

        # 有些工具可能执行成功但无文本返回
        if hasattr(result, "content") and result.content:
            return result.content[0].text

        return "工具执行成功，但无文本返回"

    async def cleanup(self):
        """关闭 MCP 服务、会话和 transport"""
        if self.session:
            await self.exit_stack.aclose()
            self.session = None
