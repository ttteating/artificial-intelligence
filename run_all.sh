#!/usr/bin/env bash
# 一键复现：切分 -> 训练终局判据 -> GA 优化 h -> 主表(dev+test) -> 可视化 -> 单测
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

echo "== 0. 依赖自检 =="
python -c "import numpy, scipy, sklearn, pandas, matplotlib, yaml" || {
  echo "缺依赖，请先: pip install -r requirements.txt"; exit 1; }

echo "== 1. 分层切分 dev/heldout/train_rest =="
python -m data.ace_loader

echo "== 2. GA 优化 h_theta（dev 适应度 + held-out 监控）=="
python -m experiments.run_ga

echo "== 3. 主表：8 个条件 × dev，随后 test 只跑一次 =="
python -m experiments.run_main_table --allow-test

echo "== 4. 搜索树可视化（成功 / 启发误导 / LLM 误判）=="
python -m experiments.run_visuals

echo "== 5. 失败案例挖掘 + h 的适应度地形扫描 =="
python -m experiments.run_failure_analysis
python -m experiments.run_hparam_sweep
python -m experiments.run_decision_eval

echo "== 6. 内核单元测试 =="
python -m tests.test_kernels

echo "== 7. 报告数字与产物一致性 =="
python -m tools.check_report_numbers

echo
echo "完成。查看： results/main_table.md / results/figs/*.png / results/trees/*.html"
