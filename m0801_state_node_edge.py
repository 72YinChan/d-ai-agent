"""
LangGraph 最小状态机

演示 LangGraph 最基础的三要素：State（状态）、Node（节点）、Edge（边）。

✅ 掌握点：
- 定义 TypedDict 作为状态（如 {"count": int}）。
- 编写节点函数（node_a, node_b）处理状态并返回更新。
- 使用 add_edge(START, "A")、add_edge("A", "B") 构建线性流程。
- 调用 app.get_graph().draw_mermaid_png() 生成可视化流程图。
输出：
- 控制台打印状态变化过程。
- 生成 workflow.png 流程图。

💡 此脚本不涉及 LLM 或对话，仅用于理解 LangGraph 底层机制。
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END, START


# 1. 定义State(状态) -- 白板上只有一个字段"count"
class State(TypedDict):
    count:int


# 2. 编写Node(节点) -- 两个"工人"
def node_a(state:State):
    # 接收当前State状态，返回要更新的部分
    print(f'[Node A]收到状态：{state}')
    return {"count": state["count"]+1}

def node_b(state:State):
    print(f'[Node B]收到状态：{state}')
    return {"count": state["count"]+1}


# 3. 添加Node到图中
workflow = StateGraph(State) # 创建画布
workflow.add_node("A", node_a) # 添加节点A
workflow.add_node("B", node_b) # 添加节点B

# 4. 用Edge连线
workflow.add_edge(START, "A") # START -> A
workflow.add_edge("A", "B") # A -> B
workflow.add_edge("B", END) # B -> END

# 编译成可运行应用
app = workflow.compile()

# 传入初始状态，执行工作流
print("---开始执行---")
result = app.invoke({"count": 1})
print("最终状态:", result)  # 输出 {'count': 3}


# 保存可视化架构图
with open('workflow.png', 'wb') as f:
    f.write(app.get_graph().draw_mermaid_png())
print("图表已保存为 workflow.png")
