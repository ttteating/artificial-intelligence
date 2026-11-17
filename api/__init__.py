"""HTTP 服务层（FastAPI）。

模块划分：
    api.schemas  —— 请求/响应模型（pydantic v2）
    api.service  —— 单例 Service：语料惰性加载、产物读取、现场搜索、解释
    api.main     —— FastAPI app 与路由
    api.__main__ —— `python -m api` 启动器
"""
