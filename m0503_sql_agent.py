"""
SQL 专用 Agent

一键构建自然语言查询数据库的智能体。

✅ 掌握点：
- 使用 create_sql_agent 快速接入 SQLite 数据库
- 自动分析表结构、生成 SQL 并执行
- 无需手动定义工具，开箱即用
"""
import os
from pathlib import Path
import sqlite3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent


load_dotenv("./.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

db_file = Path("./test_sql.db")
if db_file.exists():
    db_file.unlink()
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute("CREATE TABLE users (id INT, name TEXT, age INT);")
cursor.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30);")
cursor.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25);")
conn.commit()
conn.close()

db_uri = f"sqlite:///{db_file}"
db = SQLDatabase.from_uri(db_uri)

# 创建 SQLAgent，无需定义 tools，仅告诉它使用 openai-tools，即 Tool Calling（工具调用）模式
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
)

# 运行
response = agent_executor.invoke({"input": "告诉我Alice多大了？"})
print(response["output"])

# 清理
db._engine.dispose()  # 关闭连接池，避免文件被占用
if db_file.exists():
    db_file.unlink()
