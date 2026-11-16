"""LLM 调用成本表（B 线 Day4 交付："主表 + 分层表 + LLM 调用成本表"）。

从 `results/summary_{dev,test}.json` 读每个条件的 LLM 用量，附每场景调用、
被驳回动作、可疑缓冲区、规则剪枝与"每次调用换来的 macro-F1 增量"。

    python -m tools.llm_cost_table

产物：results/llm_cost_table.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(tag: str) -> dict:
    p = RESULTS / f"summary_{tag}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    dev, test = load("dev"), load("test")
    if not dev and not test:
        raise SystemExit("缺少 results/summary_*.json，先跑 run_main_table")

    lines = ["# LLM 调用成本表\n",
             "口径：`avg_llm_calls` 为每场景 LLM 调用次数（mock 与真实模式同一计数点），"
             "`avg_llm_rejected` 为被 `action_valid` 驳回的动作数，"
             "`avg_pruned` 为规则剪枝数。\n"]

    for tag, d in (("dev", dev), ("test", test)):
        if not d:
            continue
        base = d.get("A1", {})
        lines.append(f"## {tag}\n")
        lines.append("| 条件 | macro-F1 | 扩展节点 | LLM 调用/场景 | 驳回/场景 | 规则剪枝/场景 | "
                     "F1 − A1 | 每次调用换来的 F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cond in ("A2", "M1", "H"):
            if cond not in d:
                continue
            r = d[cond]
            delta = r["macro_f1"] - base.get("macro_f1", 0.0)
            per_call = delta / r["avg_llm_calls"] if r.get("avg_llm_calls") else float("nan")
            lines.append(f"| {cond} | {r['macro_f1']:.4f} | {r['avg_expanded']:.2f} | "
                         f"{r['avg_llm_calls']:.2f} | {r.get('avg_llm_rejected', 0):.2f} | "
                         f"{r.get('avg_pruned', 0):.2f} | {delta:+.4f} | {per_call:+.5f} |")
        if base:
            lines.append(f"| A1（无 LLM 基线） | {base['macro_f1']:.4f} | "
                         f"{base['avg_expanded']:.2f} | 0.00 | 0.00 | 0.00 | — | — |")
        lines.append("")

    lines.append("读法：\n")
    lines.append("- 如果\"每次调用换来的 F1\"接近 0 甚至为负，说明当前 LLM 评估值对最终判定没有杠杆"
                 "（见报告 §9.2 幻觉敏感性曲线：H 的 F1 在 p=0~0.5 上恒为 0.6431）。")
    lines.append("- M1 的调用量与扩展节点都是全表最高，但它与 A2 的 F1 差距在噪声内，"
                 "说明 MCTS 把预算花在了重复回访上。")
    (RESULTS / "llm_cost_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cost] -> {RESULTS / 'llm_cost_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
