"""失败案例挖掘：从已跑好的 rows CSV 里自动挑案例并附上原文。

    python -m experiments.run_failure_analysis

四类失败（对应文档"失败案例分析"交付物）：
  启发误导   A1 判错而 B0/A0 判对        —— 启发式把搜索引偏
  LLM 幻觉   H  判错而 A1 判对            —— LLM 打分/校验让搜索偏离
  规则过约束 B1 判错而 B0 判对            —— 遮挡分支导致证据不足
  MCTS 漫游  M0 判错而 A0 判对            —— 在高 reward 的相似条款上反复采样

产物：results/failure_cases.md / results/failure_cases.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.setup import ROOT

RESULTS = ROOT / "results"


def load_rows() -> pd.DataFrame:
    frames = []
    for f in ("rows_dev_all.csv", "rows_test_all.csv"):
        p = RESULTS / f
        if p.exists():
            df = pd.read_csv(p)
            df["split"] = "dev" if "dev" in f else "test"
            frames.append(df)
    if not frames:
        raise SystemExit("找不到 rows_*.csv，请先跑 experiments.run_main_table --allow-test")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_rows()
    from data.corpus import Corpus

    corpus = Corpus().load()

    df["is_correct"] = df["gold"] == df["answer"]

    def get(cond: str) -> pd.DataFrame:
        return df[df.condition == cond].set_index("sid")

    B0, B1, A0, A1, H, M0 = map(get, ("B0", "B1", "A0", "A1", "H", "M0"))

    def pick(cand: list[dict], key, n: int = 3, reverse: bool = True):
        return sorted(cand, key=key, reverse=reverse)[:n]

    mis = []
    for sid, r in A1.iterrows():
        if sid not in B0.index or sid not in A0.index:
            continue
        if (not r.is_correct) and (B0.loc[sid, "is_correct"] or A0.loc[sid, "is_correct"]):
            extra = int(r.expanded) - int(min(B0.loc[sid, "expanded"], A0.loc[sid, "expanded"]))
            mis.append({"sid": sid, "split": r.split, "gold": r.gold, "A1": r.answer,
                        "B0": B0.loc[sid, "answer"], "A0": A0.loc[sid, "answer"],
                        "A1_expanded": int(r.expanded), "extra_expanded": extra,
                        "n_clauses": int(r.n_clauses), "oracle_hops": int(r.oracle_depth)})

    llm = []
    for sid, r in H.iterrows():
        if sid in A1.index and (not r.is_correct) and A1.loc[sid, "is_correct"]:
            llm.append({"sid": sid, "split": r.split, "gold": r.gold, "H": r.answer,
                        "A1": A1.loc[sid, "answer"], "llm_calls": int(r.llm_calls),
                        "llm_rejected": int(r.llm_rejected),
                        "H_expanded": int(r.expanded)})

    rule = []
    for sid, r in B1.iterrows():
        if sid in B0.index and (not r.is_correct) and B0.loc[sid, "is_correct"]:
            rule.append({"sid": sid, "split": r.split, "gold": r.gold, "B1": r.answer,
                         "B0": B0.loc[sid, "answer"], "pruned": int(r.pruned),
                         "steps": int(r.steps), "expanded": int(r.expanded),
                         "source": "B1≠B0"})

    # 补充来源：R1~R6 规则同时进了 H / M1，若 H 因剪枝而判错（A1 对），也算"规则过约束"。
    # 只有 B1≠B0 一条渠道时，dev+test 合计常常只有 1 例，达不到文档"每类 2~3 例"。
    for sid, r in H.iterrows():
        if sid in A1.index and (not r.is_correct) and A1.loc[sid, "is_correct"] \
                and int(r.pruned) >= 2 and int(r.llm_rejected) == 0:
            rule.append({"sid": sid, "split": r.split, "gold": r.gold, "H": r.answer,
                         "A1": A1.loc[sid, "answer"], "pruned": int(r.pruned),
                         "steps": int(r.steps), "expanded": int(r.expanded),
                         "source": "H 规则剪枝且 LLM 未插手"})

    mcts = []
    for sid, r in M0.iterrows():
        if sid in A0.index and (not r.is_correct) and A0.loc[sid, "is_correct"]:
            mcts.append({"sid": sid, "split": r.split, "gold": r.gold, "M0": r.answer,
                         "A0": A0.loc[sid, "answer"], "expanded": int(r.expanded)})

    out = {
        "heuristic_misled": pick(mis, lambda x: x["extra_expanded"]),
        "llm_hallucination": pick(llm, lambda x: x["llm_calls"]),
        "rule_overconstraint": pick(rule, lambda x: x["expanded"]),
        "mcts_wander": pick(mcts, lambda x: x["expanded"]),
    }
    rule_b1 = len(rule)
    rule_h = len(rule)
    counts = {"A1≠B0 且 B0 对": len(mis), "H≠A1 且 A1 对": len(llm),
              "规则过约束（B1≠B0 + H 剪枝主导）": len(rule), "M0≠A0 且 A0 对": len(mcts)}
    if rule:
        n_b1 = sum(1 for x in rule if x.get("source") == "B1≠B0")
        counts["规则过约束（B1≠B0 + H 剪枝主导）"] = len(rule)
        print(f"[rule] B1≠B0={n_b1} / H 剪枝主导={len(rule) - n_b1}", flush=True)

    # 附原文，便于写报告
    for k, lst in out.items():
        for x in lst:
            sc = corpus.scen(x["sid"])
            x["scenario"] = sc.text[:600]
            x["clauses"] = {cid: t[:300] for cid, t in sc.clauses.items()}

    (RESULTS / "failure_cases.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    lines = ["# 失败案例（自动挖掘）\n", "## 偏离计数（dev + test 合计）\n",
             "| 现象 | 条数 |\n", "|---|---:|\n"]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |\n")
    lines.append("\n")
    for k, lst in out.items():
        lines.append(f"## {k}\n")
        if not lst:
            lines.append("_本次未挖到（说明该失效模式在本数据上出现频率极低）_\n")
            continue
        for x in lst:
            lines.append(f"### {x['sid']}（{x['split']}，gold={x['gold']}）\n")
            lines.append("```json\n" + json.dumps(
                {kk: vv for kk, vv in x.items() if kk not in ("scenario", "clauses")},
                ensure_ascii=False, indent=1) + "\n```\n")
            lines.append(f"- 场景摘录：{x['scenario'][:260]}…\n")
            for cid, t in list(x["clauses"].items())[:3]:
                lines.append(f"- 条款 `{cid}`：{t[:180]}…\n")
            lines.append("")
    (RESULTS / "failure_cases.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines[:40]))
    print(f"[cases] -> {RESULTS/'failure_cases.md'}")


if __name__ == "__main__":
    main()
