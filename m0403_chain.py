"""
链式调用

将 Prompt + LLM + Parser 组合成 Chain。

✅ 掌握点：
- 使用 | 操作符连接组件
- 创建可复用的处理流程
- 输出解析为字符串
"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


prompt = ChatPromptTemplate.from_messages([
    ("system", "你非常可爱，说话末尾会带个喵"),
    ("human", "{input}"),
])

llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"input": "你好喵"})
print(result)
