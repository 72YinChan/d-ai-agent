"""
模型调用

封装 LLM 实例，实现标准调用流程。

✅ 掌握点：
- 如何初始化 ChatOpenAI 或 DeepSeek
- 设置 API Key 和 base_url
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

response = llm.invoke("你好喵")
print(response.content)
