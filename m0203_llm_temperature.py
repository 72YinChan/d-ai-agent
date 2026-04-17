"""
温度参数对输出的影响

演示不同 temperature 参数对 LLM 输出多样性和创造性的影响。

🔧 使用提示：
想测试不同温度参数的效果？
👉 直接修改代码中的 temperature 值即可（范围通常为 0-2）。

✅ 掌握点：
- temperature 参数的作用和影响
- 如何根据场景选择合适的温度值
- 低温度（稳定）与高温度（随机）输出的对比
"""
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
