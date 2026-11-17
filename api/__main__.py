"""`python -m api` 启动器。

    python -m api                     # 127.0.0.1:8000
    python -m api --port 8010 --reload
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="启动合同合规搜索服务（FastAPI）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="开发模式：改动代码自动重启")
    args = ap.parse_args()

    import uvicorn

    if args.reload:
        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run("api.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
