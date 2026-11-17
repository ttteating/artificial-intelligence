"""加载项目根目录的 .env（可选依赖 python-dotenv）。

只补全缺失的环境变量，不覆盖已存在的值，优先级：
进程环境变量 > .env > 代码默认值。
python-dotenv 未安装时静默跳过，保持原有 `set LLM_API_KEY=...` 用法可用。
"""

from __future__ import annotations

from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_loaded = False


def load_project_env() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_PATH, override=False)
