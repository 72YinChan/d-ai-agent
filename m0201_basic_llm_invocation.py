import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},  # 提示词角色
        {"role": "user", "content": "Hello"},  # 用户输入的对话
    ],
    stream=False,  # 非流式输出，只会等语句全部生成才返回
)
print(response.choices[0].message.content)
