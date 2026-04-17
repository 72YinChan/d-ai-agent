"""
自定义函数调用

直接调用该函数，运行成功即可，重点掌握： ✅ 了解Agent概念 ✅ 成功运行该代码
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

response = llm.invoke("你好呀")
print(response.content)
