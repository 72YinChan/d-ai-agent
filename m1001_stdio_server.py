"""
标准输入输出MCP服务器

实现基于标准输入输出的MCP服务器，支持本地进程间通信。

✅ 掌握点：
- 使用FastMCP创建MCP服务实例
- 定义和注册工具函数
- 异步工具函数的实现方式
- Stdio通信模式的配置与运行
功能：
- 创建名为"WeatherService"的MCP服务器
- 提供get_weather工具，模拟查询指定城市天气
- 支持异步调用，FastMCP自动处理协程
- 采用标准输入输出作为通信通道

💡 这种模式适合本地进程间通信，程序启动后会等待客户端指令，无默认输出。
"""
from mcp.server.fastmcp import FastMCP


# 初始化服务
# WeatherService 是服务的名字
mcp = FastMCP("WeatherService")

# 业务逻辑工具
@mcp.tool()
async def get_weather(city: str):
    """
    查询指定城市的实时天气。
    如果是此时此刻的天气请求，调用此工具。
    """
    # 模拟真实的网络请求（可以替换成自己的天气 API）
    # 在 MCP 中，工具函数可以是 async 的，FastMCP 会自动处理。
    return f"{city}的天气是：晴，气温25℃，风力3级"


if __name__ == "__main__":
    # 默认运行方式：stdio（标准输入输出）
    # 这种模式下，程序启动后会“挂起”等待指令，不会有任何打印输出。
    mcp.run()
