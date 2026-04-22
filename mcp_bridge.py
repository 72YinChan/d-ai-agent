from typing import Dict, Any, List
from contextlib import AsyncExitStack

from pydantic import Field, create_model
from langchain_core.tools import StructuredTool

from mcp_client import MCPClient


class LangChainMCPAdapter:
    """
    MCP 适配器：将 MCP 客户端无缝转换为 LangChain 可用的工具集。
    实现了上下文管理器协议
    """
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client

    async def __aenter__(self):
        """进入上下文，自动建立连接"""
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        """退出上下文，自动清理资源"""
        await self.client.cleanup()

    @staticmethod
    def _schema_to_pydantic(name: str, schema: Dict[str, Any]):
        """
        将 MCP 的 JSON Schema 动态转换为 Pydantic 模型
        这是让 LLM 理解参数要求的关键
        """
        # print(f"🔧 调试：工具 '{name}' 的 inputSchema = {schema}")  # 查看 MCP 返回的原始 inputSchema

        # 所有参数定义
        properties = schema.get("properties", {})  # 允许为空
        # 必须字段
        required = schema.get("required", [])  # 允许为空

        # 初始空字典
        fields = {}

        # 类型映射表：将 JSON 类型映射为 Python 类型
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List,
            "object": Dict,
        }

        for field_name, field_info in properties.items():
            # 1. 获取字段类型
            json_type = field_info.get("type", "string")
            python_type = type_map.get(json_type, Any)

            # 2. 获取描述
            description = field_info.get("description", "")

            # 3. 是否为必需项
            # 如果是必填，默认值为 ... (Ellipsis): 否则为 None
            if field_name in required:
                default_value = ...
            else:
                default_value = None

            # 4. 构建 Pydantic 字段定义 —— create_model 要求的特定格式
            fields[field_name] = (python_type, Field(default=default_value, description=description))

        # 动态创建一个 Pydantic 模型类
        return create_model(f"{name}Schema", **fields)

    async def get_tools(self):
        """
        核心方法：获取并转换工具
        返回的是标准的 LangChain Tool 列表，可以直接喂给 bind_tools
        """
        # 从 MCP Server 获取原始工具列表
        mcp_tools = await self.client.list_tools()
        langchain_tools = []

        for tool_info in mcp_tools:
            # 1. 动态生成参数模型 —— 要处理 schema 为空的情况
            # inputSchema 一般会放好 MCP 各种工具/参数的介绍
            raw_schema = tool_info.get("input_schema", {})
            args_model = self._schema_to_pydantic(tool_info["name"], raw_schema)
            # 2. 定义执行函数
            async def _dynamic_tool_func(tool_name=tool_info["name"], **kwargs):
                # ⚠️: 必须绑定 tool_name 默认参数，否则循环会覆盖 tool_name
                return await self.client.call_tool(tool_name, kwargs)

            # 3. 包装成 llm 可调用的工具（注入 args_schema）
            tool = StructuredTool.from_function(
                coroutine=_dynamic_tool_func,
                name=tool_info["name"],
                description=tool_info["description"],
                args_schema=args_model,  # 把说明书传给 LangChain
            )
            langchain_tools.append(tool)
        return langchain_tools

    @classmethod
    async def load_mcp_tools(cls, stack: AsyncExitStack, configs: list):
        """
        负责遍历配置，批量建立连接，收集所有工具
        使用 stack 将连接生命周期托管给上层
        """
        all_tools = []
        for conf in configs:
            print(f"🔌 正在连接: {conf['name']} == ({conf.get('transport', 'stdio')})...")

            # 根据 transport 类型创建不同的客户端
            transport = conf.get("transport", "stdio")
            if transport == "stdio":
                # 初始化 Client
                client = MCPClient(
                    transport="stdio",
                    command=conf["command"],
                    args=conf["args"],
                    env=conf.get("env"),  # 可选参数
                )
            else:  # http
                client = MCPClient(
                    transport="http",
                    url=conf["url"],
                )

            # 🔥: enter_async_context 替代了 async with 缩进
            # 这样无论有多少个 MCP，代码层级都不会变深
            adapter = await stack.enter_async_context(cls(client))
            # 批量获取一个 MCP 下的所有工具
            tools = await adapter.get_tools()
            print(f"    ✅️ 获取工具{[t.name for t in tools]}")
            all_tools.extend(tools)

        return all_tools
