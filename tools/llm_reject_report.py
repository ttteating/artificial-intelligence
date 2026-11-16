"""LLM 误杀率统计（文档 §2.3 Day5：B 线要准备"LLM 误杀率"）。

定义
----
**误杀** = LLM 的 `action_valid` 把一个"本来正确"的迁移判成非法。判定"本来正确"的
口径：该 (src, edge_type, dst) 出现在 **A1（无 LLM 的 A* + GA）判对时的成功路径**上。
所以分母是 A2 / H / M1 在 dev 上的全部 LLM 驳回次数，分子是其中落在 A1 正确路径上的次数。

    python -m tools.llm_reject_report              # dev 600
    python -m tools.llm_reject_report --limit 300

产物：results/llm_reject_rate.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.factory import build_problem, run_kernel
from experiments.setup import prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def reference_paths(corpus, sids, ctx) -> dict[str, set]:
    """跑 A1，把"判对"场景的成功路径当参照（无 LLM 干扰）。"""
    refs = {}
    for sid in sids:
        problem, priority, spec = build_problem(corpus, sid, "A1", ctx)
        res = run_kernel(problem, priority, spec, ctx)
        if res.answer == corpus.scen(sid).label:
            refs[sid] = {(a.src, a.edge_type, a.dst) for a in res.path}
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = dev 全量")
    args = ap.parse_args()

    ctx = prepare(quiet=False)
    sids = ctx.corpus.ids("dev")
    if args.limit:
        import random
        rng = random.Random(ctx.seed * 7919 + args.limit)
        sids = sorted(rng.sample(sids, min(args.limit, len(sids))))

    print(f"[reject] 先跑 A1 求参照路径（{len(sids)} 条 dev）…", flush=True)
    refs = reference_paths(ctx.corpus, sids, ctx)

    rows = []
    for cond in ("A2", "H", "M1"):
        total_veto = killed = 0
        susp = 0
        n = 0
        for sid in sids:
            problem, priority, spec = build_problem(ctx.corpus, sid, cond, ctx)
            res = run_kernel(problem, priority, spec, ctx)
            veto = list(getattr(problem.rules, "veto_log", []) or [])
            susp += int(getattr(res, "llm_suspicious", 0) or 0)
            total_veto += len(veto)
            if sid in refs:
                killed += sum(1 for v in veto
                              if (v["src"], v["edge_type"], v["dst"]) in refs[sid])
            n += 1
        rate = (killed / total_veto) if total_veto else 0.0
        rows.append({"condition": cond, "scenarios": n,
                     "llm_vetoes": total_veto, "false_kills": killed,
                     "false_kill_rate": rate, "suspicious": susp})
        print(f"  [{cond}] vetoes={total_veto} false_kills={killed} "
              f"rate={rate:.2%} suspicious={susp}", flush=True)

    md = ["# LLM 误杀率统计（dev）\n",
          f"- 参照：A1（无 LLM）判对场景的成功路径，共 {len(refs)}/{len(sids)} 条场景",
          "- 误杀 = LLM `action_valid` 驳回的迁移中，落在参照路径上的比例",
          "- 同时统计\"可疑缓冲区\"（合法但置信度 < 0.40 而被后置的动作数）\n",
          "| 条件 | 场景数 | LLM 驳回次数 | 误杀次数 | **误杀率** | 可疑缓冲区动作数 |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['condition']} | {r['scenarios']} | {r['llm_vetoes']} | "
                  f"{r['false_kills']} | {r['false_kill_rate']:.2%} | {r['suspicious']} |")
    md.append("")
    md.append("> 说明：`--llm mock` 下驳回来自 `MockLLMEvaluator` 的规则模拟"
              "（目标条款相似度 < 0.08 判非法），因此这里的误杀率是**评估器口径**的证据，"
              "换真实模型后同一条脚本可直接复用。")
    (RESULTS / "llm_reject_rate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / "llm_reject_rate.json").write_text(
        json.dumps({"reference": len(refs), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[reject] -> {RESULTS / 'llm_reject_rate.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
