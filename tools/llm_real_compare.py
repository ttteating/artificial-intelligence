"""真实 LLM 与 mock 在同一子集上的逐条对照。

用途：`--llm real` 只跑小样本时，必须用同一批场景再跑一次 `--llm mock`，
否则"真实 vs mock"的差值会混入子集差异。本脚本按 sid 对齐两份 rows 文件，
输出指标差、判定翻转条数与每条翻转的明细。

用法：
    python -m experiments.run_baselines --condition H --split dev --limit 20 --llm mock --out results/rows_H_dev_mock20.json
    python -m experiments.run_baselines --condition H --split dev --limit 20 --llm real --out results/rows_H_dev_real20.json
    python -m tools.llm_real_compare --mock results/rows_H_dev_mock20.json --real results/rows_H_dev_real20.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["sid"]: r for r in rows}


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def score_block(rows: list[dict]) -> dict:
    gold = [r["gold"] for r in rows]
    pred = [r["answer"] for r in rows]
    total_expanded = sum(r["expanded"] for r in rows) or 1
    return {
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "acc": sum(g == p for g, p in zip(gold, pred)) / len(rows),
        "expanded": mean(rows, "expanded"),
        "steps": mean(rows, "steps"),
        "redundant": sum(r["redundant"] for r in rows) / total_expanded,
        "llm_calls": mean(rows, "llm_calls"),
        "llm_rejected": sum(r["llm_rejected"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_real_vs_mock.md"))
    args = ap.parse_args()

    mock = load(args.mock)
    real = load(args.real)
    sids = sorted(set(mock) & set(real))
    if not sids:
        raise SystemExit("[compare] 两份 rows 没有共同 sid")

    m_rows = [mock[s] for s in sids]
    r_rows = [real[s] for s in sids]
    m, r = score_block(m_rows), score_block(r_rows)

    flip = [s for s in sids if mock[s]["answer"] != real[s]["answer"]]
    mock_only = [s for s in flip if mock[s]["answer"] == mock[s]["gold"]]
    real_only = [s for s in flip if real[s]["answer"] == real[s]["gold"]]
    both_wrong = [s for s in flip if mock[s]["answer"] != mock[s]["gold"] and real[s]["answer"] != real[s]["gold"]]

    lines = [
        "# 真实 LLM vs mock（同一子集逐条对照）",
        "",
        f"- 场景数：{len(sids)}（按 sid 对齐，两份文件交集）",
        f"- mock：`{args.mock}`",
        f"- real：`{args.real}`",
        "",
        "| 指标 | mock | real | 差（real − mock） |",
        "|---|---:|---:|---:|",
        f"| macro-F1 | {m['macro_f1']:.4f} | {r['macro_f1']:.4f} | {r['macro_f1'] - m['macro_f1']:+.4f} |",
        f"| accuracy | {m['acc']:.4f} | {r['acc']:.4f} | {r['acc'] - m['acc']:+.4f} |",
        f"| 平均扩展节点 | {m['expanded']:.2f} | {r['expanded']:.2f} | {r['expanded'] - m['expanded']:+.2f} |",
        f"| 平均推理步数 | {m['steps']:.2f} | {r['steps']:.2f} | {r['steps'] - m['steps']:+.2f} |",
        f"| 冗余占比 | {m['redundant']:.3f} | {r['redundant']:.3f} | {r['redundant'] - m['redundant']:+.3f} |",        f"| LLM 调用/场景 | {m['llm_calls']:.1f} | {r['llm_calls']:.1f} | {r['llm_calls'] - m['llm_calls']:+.1f} |",
        f"| LLM 驳回总数 | {m['llm_rejected']} | {r['llm_rejected']} | {r['llm_rejected'] - m['llm_rejected']:+d} |",
        "",
        f"判定翻转：{len(flip)} / {len(sids)} 条（{len(flip) / len(sids):.1%}）",
        f"- 只有 mock 对：{len(mock_only)} 条",
        f"- 只有 real 对：{len(real_only)} 条",
        f"- 两边都错：{len(both_wrong)} 条",
        "",
    ]

    if flip:
        lines += [
            "| sid | gold | mock 判定 | real 判定 | mock 扩展 | real 扩展 | real 驳回 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for s in flip:
            lines.append(
                f"| {s} | {mock[s]['gold']} | {mock[s]['answer']} | {real[s]['answer']} | "
                f"{mock[s]['expanded']} | {real[s]['expanded']} | {real[s]['llm_rejected']} |"
            )
        lines.append("")

    lines += [
        "## 读法",
        "",
        "1. 子集只有 20 条，单条翻转就是 5 个点的 F1 摆动，**不能当统计结论**；这里只证明两件事：",
        "   真实模型能被搜索链正常调用、且它的判定与 mock 确实不同（不是空转）。",
        "2. `LLM 驳回总数` 是动作级计数（同一动作被驳回多次计多次）。真实模型这一列通常更高：",
        "   它对中间状态措辞更挑剔；而 `LLM 调用/场景` 有磁盘缓存去重（相同 prompt 不重复计费），",
        "   所以两列不能互相替代——调用少不等于校验松。",
        "3. 要给出\"真实 LLM 是否优于 mock\"的结论，至少需要 200 例以上（见报告 §9.1）。",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
