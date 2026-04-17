"""
提示词构建

使用 ChatPromptTemplate 构建可复用的 Prompt 模板。

✅ 掌握点：
- 系统提示 + 历史消息 + 用户输入
- 使用 {input} 占位符
- 多轮对话的基础结构
"""
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages([
    ("system", "你非常可爱，说话末尾会带个喵"),
    ("human", "{input}"),
])
# 格式化输出
formatted_prompt = prompt.invoke({"input": "你好喵"})
print(formatted_prompt)
