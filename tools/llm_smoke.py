"""真实 LLM 通路冒烟（3 次调用）：确认 adapter 能连上、能返回合法 JSON。

为什么需要这一步：主表的 A2 / M1 / H 在无 `LLM_API_KEY` 时会静默回退 mock，
`llm_mode` 仍是 "mock"，报告里所有 LLM 数字都只能当"接口形状"。本脚本用
**最小代价**（2 次 action_valid + 1 次 path_reward）验证真实通路，再决定是否
值得花 token 跑全量。

用法（任选其一）：
    set LLM_API_KEY=sk-...
    python -m tools.llm_smoke
    python -m experiments.run_baselines --condition H --split dev --limit 20 --llm real

    python -m tools.llm_smoke --key sk-... --base-url https://api.deepseek.com/v1 --model deepseek-chat

产物：results/llm_smoke.md（含 3 条返回原文与耗时、是否命中标签词）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="", help="不设环境变量时可直接传 key")
    ap.add_argument("--base-url", default="", help="OpenAI 兼容网关，如 https://api.deepseek.com/v1")
    ap.add_argument("--model", default="", help="模型名，默认读 LLM_MODEL 或 gpt-4o-mini")
    ap.add_argument("--sid", default="", help="可选：用真实场景文本构题")
    args = ap.parse_args()

    if args.key:
        os.environ["LLM_API_KEY"] = args.key
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    from llm.adapter import LLMClient, REWARD_SYS, VALIDATOR_SYS

    cli = LLMClient()
    if not cli.available:
        print("[smoke] 没有 LLM_API_KEY，无法跑真实通路；请先设置 key 或传 --key。")
        return 2

    print(f"[smoke] model={cli.model} base_url={cli.base_url or 'default'} "
          f"cache={cli.cache_dir}")

    ctx = ("SCENARIO:\nThe franchisee intends to transfer the franchise without paying "
           "outstanding royalties. The agreement requires payment in full before transfer.\n\n"
           "FROM:\nIf you propose to Transfer this Agreement, you must first obtain our consent.\n\n"
           "MOVE: exception_of\nTO:\nNotwithstanding the foregoing, no consent is required "
           "if all outstanding amounts have been paid.")

    rows = []
    for tag, sys_p, user in (("valid-A", VALIDATOR_SYS, ctx),
                             ("valid-B", VALIDATOR_SYS, ctx.replace("no consent is required",
                                                                    "consent is always required")),
                             ("reward", REWARD_SYS, ctx)):
        t0 = time.time()
        try:
            obj = cli.json_call(sys_p, user, tag=tag)
            err = ""
        except Exception as exc:                       # noqa: BLE001
            obj, err = {}, f"{type(exc).__name__}: {exc}"
        rows.append({"tag": tag, "obj": obj, "error": err,
                     "elapsed_s": round(time.time() - t0, 2)})
        print(f"  [{tag}] {json.dumps(obj, ensure_ascii=False)} {err} "
              f"({rows[-1]['elapsed_s']}s)")

    md = ["# 真实 LLM 通路冒烟\n",
          f"- model: `{cli.model}`；base_url: `{cli.base_url or '默认'}`",
          f"- 真实请求次数: {cli.calls}；缓存命中: {cli.cache_hits}",
          "- 每次调用的 `violation` 为 true 表示模型输出了合规标签词，分数已被清零\n",
          "| 调用 | 返回 | 耗时(s) | 错误 |", "|---|---|---:|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | `{json.dumps(r['obj'], ensure_ascii=False)}` | "
                  f"{r['elapsed_s']} | {r['error'] or '—'} |")
    md.append("")
    md.append("通过标准：三条都返回 JSON、`valid`/`reward` 字段可解析、没有 `violation`。")
    md.append("通过后再跑 `python -m experiments.run_baselines --condition H --split dev "
              "--limit 20 --llm real` 验证接到搜索链上。")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "llm_smoke.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[smoke] -> {RESULTS / 'llm_smoke.md'}")
    return 0 if all(not r["error"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
