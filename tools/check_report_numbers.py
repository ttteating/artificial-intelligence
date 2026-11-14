"""报告数字一致性检查：把报告正文引用的数字与 results/ 产物逐个比对。

报告开头写了"本文中的每个数字都可以在 results/ 下找到对应产物"，这个脚本就是那句话的执行器：
数值被改了但报告没同步（或反过来）时，它会列出缺漏，避免"报告里的旧数字"这种最伤可信度的问题。

用法：python -m tools.check_report_numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "报告_成员A_搜索算法线.md"
CONDS = ("B0", "B1", "A0", "A1", "A2", "M0", "M1", "H")


def load(p: str):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main() -> int:
    if not REPORT.exists():
        print("找不到报告：", REPORT)
        return 1
    rep = REPORT.read_text(encoding="utf-8")
    missing: list[tuple[str, str]] = []

    def need(token: str, label: str) -> None:
        if token not in rep:
            missing.append((label, token))

    # ---- 主表（dev / test）
    for tag, f in (("dev", "results/summary_dev.json"), ("test", "results/summary_test.json")):
        d = load(f)
        for c in CONDS:
            v = d[c]
            need(f"{v['macro_f1']:.4f}", f"{tag}/{c} macro-F1")
            need(f"{v['avg_expanded']:.2f}", f"{tag}/{c} 平均扩展")
            need(f"{v['avg_steps']:.2f}", f"{tag}/{c} 推理步数")

    # ---- 预算扫描（报告里写成 .6354 这种前导小数点形式）
    for _, r in pd.read_csv(ROOT / "results" / "budget_sweep.csv").iterrows():
        need(f"{r['macro_f1']:.4f}"[1:], f"budget={r['budget']} {r['condition']}")

    # ---- 判据单独评估
    dp = load("results/decision_path_eval.json")
    need(f"{dp['heldout']['macro_f1']:.4f}", "判据 held-out")
    need(f"{dp['dev']['macro_f1']:.4f}", "判据 dev")

    # ---- GA
    ga = load("results/ga_history.json")["summary"]
    need(f"{ga['dev_macro_f1']:.4f}", "GA dev")
    need(f"{ga['heldout_macro_f1']:.4f}", "GA held-out")
    for k, v in ga["best_params"].items():
        need(f"{v:.2f}".rstrip("0").rstrip("."), f"GA {k}")

    # ---- h 参数地形
    hs = pd.read_csv(ROOT / "results" / "hparam_sweep.csv")
    for b, g in hs.groupby("budget"):
        need(f"{g['macro_f1'].std():.4f}", f"hparam b={b} std")
        need(f"{g['macro_f1'].max():.4f}", f"hparam b={b} max")

    if missing:
        print(f"报告与产物不一致：{len(missing)} 处")
        for label, tok in missing:
            print(f"  - {label}: 期望出现 {tok!r}")
        return 1

    print("报告数字与 results/ 产物一致 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
