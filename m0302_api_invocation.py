import os

from dotenv import load_dotenv
from openai import OpenAI
import requests


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


def create_client():
    return OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)


def get_attr():
    res_ip = requests.get("https://ipapi.co/ip/").text
    res_city = requests.get("https://ipapi.co/city/").text
    return f"IP地址: {res_ip}, 所在城市: {res_city}"


get_addr_func = {
    "name": "get_addr",
    "description": "获取用户的ip地址与城市",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    }
}

tools = [
    {
        "type": "function",
        "function": get_addr_func,
    }
]

def chat_loop(agent_client, tools):
    messages = [
        {"role": "system", "content": "你是一个善解人意会热心回答人问题的助手。如果你感觉你回答不了当前问题，就会调用函数来回答。"},
        {"role": "user", "content": "我现在在哪个城市，ip地址是多少?"},
    ]
    response = agent_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    message = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_addr":
                # 这里无参数，直接调用函数就行
                your_info = get_attr()
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": your_info,
                })
                final_res = agent_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                print("已调用工具...")
                print(f"回答:{final_res.choices[0].message.content}")
    else:
        print("未调用工具...")
        print(f"回答:{response.choices[0].message.content}")


if __name__ == "__main__":
    client = create_client()
    chat_loop(client, tools)
