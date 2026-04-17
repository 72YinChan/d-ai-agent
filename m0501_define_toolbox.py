"""
工具函数定义

使用 @tool 装饰器将 Python 函数封装为 LangChain 可调用的工具。

✅ 掌握点：
- 如何用 @tool 定义工具
- 必须添加三引号描述（作为提示词）
- 支持任意自定义逻辑函数
"""
from langchain_core.tools import tool


@tool
def get_weather(location):
    """模拟获取天气信息"""
    return f"{location}当前天气：23℃，晴，风力2级"


@tool
def get_user_name(user):
    """模拟获取用户名"""
    return f"用户名为：{user}"


tools = [get_weather, get_user_name]
print("工具箱封装完毕！")
