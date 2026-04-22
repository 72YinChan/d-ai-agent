"""
传输层协议接口

定义MCP传输层的抽象协议接口，为所有传输实现提供统一的规范。

✅ 掌握点：
- Python Protocol的使用方法
- 抽象接口的设计原则
- MCP协议的核心方法定义
功能：
- 定义MCP传输层必须实现的四个核心方法：connect、list_tools、call_tool、cleanup
- 提供类型注解，确保接口一致性
- 为不同传输实现提供统一的调用方式

💡 这是整个组件系统的基础，定义了传输层的契约，使得上层代码可以与具体传输实现解耦。
"""
from typing import Protocol, Dict


class MCPTransport(Protocol):
    """MCP 传输层协议 —— 所有 transport 必须实现如下方法"""
    async def connect(self):
        """建立连接"""
        ...

    async def list_tools(self):
        """获取工具列表"""
        ...

    async def call_tool(self, name: str, args: Dict):
        """调用工具并返回文本结果"""
        ...

    async def cleanup(self):
        """清理资源"""
        ...
