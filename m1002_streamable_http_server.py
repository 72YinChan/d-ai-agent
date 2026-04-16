"""
HTTP协议MCP服务器

实现基于HTTP协议的MCP服务器，支持远程网络调用。

✅ 掌握点：
- FastMCP服务器的网络配置
- HTTP通信模式的实现
- 监听地址和端口的设置
- Streamable HTTP运行模式的使用
功能：
- 创建名为"WeatherService"的网络MCP服务器
- 配置监听地址(0.0.0.0)和端口(8001)
- 提供与stdio_server相同的get_weather工具
- 自动启动uvicorn服务器，支持远程调用

💡 这种模式适合构建可远程访问的MCP服务，便于分布式系统集成。
"""
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("WeatherService", host="0.0.0.0", port=8001)

# 业务逻辑工具
@mcp.tool()
async def get_weather(city: str):
    """
    查询指定城市的实时天气
    如果是此时此刻的天气请求，调用此工具。
    """
    # 模拟真实的网络请求（可以替换成自己的天气 API）
    # 在 MCP 中，工具函数可以是 async 的，FastMCP 会自动处理
    return f"{city}的天气是：晴，气温25℃，风力3级"


if __name__ == "__main__":
    # 运行方式：Streamable HTTP
    # 这会自动启动 uvicorn 服务器，支持远程调用
    mcp.run("streamable-http")
