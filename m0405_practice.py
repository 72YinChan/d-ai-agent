"""
综合实践

整合所有组件，搭建一个带记忆的 AI 对话机器人。

✅ 掌握点：
- 完整流程串联
- 实际运行体验
- 可直接扩展为 Web 应用
"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


prompt = """\
你是‘小智’，一位专业、耐心且记忆力出色的 AI 助手。
你善于倾听，能记住用户之前提到的信息，并在后续对话中自然提及。
回答时简洁明了，避免冗余。
"""


llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)


def create_bot(llm, sys_prompt):
    """
    :param llm: 已配置好的语言模型实例
    :param sys_prompt: 系统提示词
    :return:
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser
    store = {}

    def get_session_history(session_id):
        return store.setdefault(session_id, ChatMessageHistory())

    return RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def main():
    # 此处集中配置LLM
    bot = create_bot(llm, prompt)
    session_id = "123"
    while True:
        user_input = input("\n你: ")
        if user_input == "quit":
            print("拜拜")
            break
        response = bot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print("AI: ", response)


if __name__ == "__main__":
    main()
