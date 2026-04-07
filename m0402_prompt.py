from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages([
    ("system", "你非常可爱，说话末尾会带个喵"),
    ("human", "{input}"),
])
# 格式化输出
formatted_prompt = prompt.invoke({"input": "你好喵"})
print(formatted_prompt)
