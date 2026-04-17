"""
文本向量化功能

项目使用根目录下的 embeddings.py 模块提供文本向量化功能，该模块封装了 HuggingFaceEmbeddings，支持自动下载和加载本地 Embedding 模型。

✅ 掌握点：
- 通过 get_embeddings() 函数获取向量化模型，默认使用 BAAI/bge-small-zh-v1.5。
- 函数会自动下载模型到本地 models 目录，之后可离线使用。
- 理解 Embedding（向量化）是 RAG 的"语义标签"，用于实现"语义搜索"。
"""
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download


def get_embeddings(model_name="BAAI/bge-small-zh-v1.5", device="cpu", **kwargs):
    # 支持更换其他向量化模型
    local_dir = Path("models") / model_name.replace("/", "_")
    if not local_dir.exists():
        print(f"⚠️ 首次加载 Embedding 模型，正在下载到{local_dir.absolute()}")
        print("💡 提示：需要联网(必需梯子)，完成后可离线使用")
        # 模型下载工具
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        print("✅ 下载完成！")

    # 构造参数字典
    model_kwargs = {
        "device": device,
        "local_files_only": True,  # 仅使用本地文件
    }

    # 此处实例化时，把 kwargs 传入
    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=str(local_dir),  # 使用本地已下载的模型
        model_kwargs=model_kwargs,
        **kwargs,  # 允许传入参数
    )
    return _EMBEDDINGS
