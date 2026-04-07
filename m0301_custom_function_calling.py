import os
import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


def create_client():
    return OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)


def get_weather(location):
    return f"{location}当前天气：23℃，晴，风力2级"


get_weather_func = {
    "name": "get_weather",
    "description": "获取指定城市的天气情况",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名称，如北京、上海等",
            },
        },
        "required": ["location"],
    }
}

tools = [
    {
        "type": "function",
        "function": get_weather_func,
    }
]

def chat_loop(agent_client, tools):
    messages = [
        {"role": "system", "content": "你是一个善解人意会热心回答人问题的助手。如果你感觉你回答不了当前问题，就会调用函数来回答。"},
        {"role": "user", "content": "北京今天的天气怎么样？"},
    ]
    response = agent_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,  # 调用工具
        tool_choice="auto",  # 模型自主选择是否调用工具
    )
    message = response.choices[0].message
    if message.tool_calls:  # 如果有该参数，证明ai调用了工具
        # 对每个可能要调用的工具进行循环
        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_weather":
                # 解析参数
                args = json.loads(tool_call.function.arguments)  # 获取用户关键词的参数
                location = args.get("location", "未知地点")

                # 调用真实参数
                weather_info = get_weather(location)

                # 将函数执行结果以"tool"角色传给模型，等待后面二次调用
                messages.append(message)  # 先添加模型的原始响应
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": weather_info,
                })
        # 第二次调用，让模型基于工具返回的结果再生成最终答案
        final_res = agent_client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        print("已调用工具...")
        print(f"回答:{final_res.choices[0].message.content}")
    else:
        # 模型没有要调用工具, 直接返回
        print("未调用工具...")
        print(f"回答:{response.choices[0].message.content}")


if __name__ == "__main__":
    client = create_client()
    chat_loop(client, tools)
