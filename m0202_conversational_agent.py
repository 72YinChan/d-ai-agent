"""
多轮问答 Agent

整合了 LLM + Prompt + Memory，实现有记忆的对话。

🔧 使用提示：
想换 AI 人设或初始提问？
👉 直接修改 messages 中的 content 即可。

✅ 掌握点：
- 多轮对话如何保持上下文
- Prompt 结构设计
- 记忆模块（Memory）的基本用法
"""
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


def create_client():
    return OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)


def chat_loop(agent_client):
    messages = [
        {"role": "system", "content": "你是一个历史老师，你会耐心的教导我有关的事情。同时你的回答会相对精简，在五十字内。"},
        {"role": "user", "content": "汉朝存在了多久，其中哪个皇帝你认为最厉害？"},
    ]
    while True:
        response = agent_client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        answer = response.choices[0].message.content
        print(f"回答:{answer}")

        user_input = input("您还有其他想继续问的吗 | (exit退出)\n")
        if user_input == "exit":
            break

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    client = create_client()
    chat_loop(client)
