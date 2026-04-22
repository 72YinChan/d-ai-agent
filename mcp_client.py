from typing import Literal, Optional, List, Dict
import uuid
from contextlib import AsyncExitStack

from .transports.base import MCPTransport
from .transports.stdio import StdioMCPTransport
from .transports.http import HttpMCPTransport


class MCPClient:

    # 编辑器 _impl 必须满足 MCPTransport 协议
    _impl: MCPTransport

    def __init__(
        self,
        transport: Literal["stdio", "http"] = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict] = None,
        url: Optional[str] = None,
    ):
        """
        MCP 客户端 - 支持 stdio 与 HTTP 两种传输方式

        :param transport: 传输模式 "stdio" / "http"
        :param command: stdio 模式的命令（如 npx）
        :param args: stdio 模式的参数
        :param env: stdio 模式的环境变量
        :param url: http 模式的端点 URL
        """
        if transport == "stdio":
            if command is None:
                raise ValueError("stdio 传输模式需要参数 'command'")
            self._impl = StdioMCPTransport(command=command, args=args or [], env=env)
        elif transport == "http":
            if url is None:
                raise ValueError("http 传输模式需要参数 'command'")
            self._impl = HttpMCPTransport(url=url)
            self.url = url
        else:
            raise ValueError(f"不支持的传输模式: {transport}")

    async def connect(self):
        """建立 MCP 连接（stdio 或 HTTP）"""
        await self._impl.connect()

    async def list_tools(self):
        """查询工具列表，为 LLM 建立上下文用"""
        return await self._impl.list_tools()

    async def call_tool(self, name: str, args: Dict):
        """调用工具（工程化：加上防御性处理）"""
        return await self._impl.call_tool(name, args)

    async def cleanup(self):
        """关闭 MCP 服务、会话和 transport"""
        return await self._impl.cleanup()

    async def _http_request(self, method: str, params: Optional[Dict] = None):
        """发送 HTTP JSON-RPC 请求"""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        response = await self.http_client.post(
            self.url,
            json=payload,
            headers=headers,
        )

        if "Mcp-Session-Id" in response.headers:
            self.session_id = response.headers["Mcp-Session-Id"]

        response.raise_for_status()
        return response.json()
