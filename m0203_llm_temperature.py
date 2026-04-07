import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

response_low_temperature = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个科学解说员，请用生动形象的语言回答问题。"},
        {"role": "user", "content": "请用一句话描述量子力学的奇妙之处。"},
    ],
    temperature=0.1,
)

response_high_temperature = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个科学解说员，请用生动形象的语言回答问题。"},
        {"role": "user", "content": "请用一句话描述量子力学的奇妙之处。"},
    ],
    temperature=1.3,
)

# 测试1：低温度 (稳定)
print(f"温度 0.1: {response_low_temperature.choices[0].message.content}")

# 测试2：高温度 (随机)
print(f"温度 1.3: {response_high_temperature.choices[0].message.content}")
