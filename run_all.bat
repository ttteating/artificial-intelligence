@echo off
chcp 65001 >nul
cd /d %~dp0
set PYTHONIOENCODING=utf-8
echo == 0. 依赖自检 ==
python -c "import numpy, scipy, sklearn, pandas, matplotlib, yaml" || goto :err
echo == 1. 分层切分 dev / heldout / train_rest ==
python -m data.ace_loader || goto :err
echo == 2. GA 优化 h_theta，dev 适应度 + held-out 监控 ==
python -m experiments.run_ga || goto :err
echo == 3. 主表 8 条件 x dev，随后 test 只跑一次 ==
python -m experiments.run_main_table --allow-test || goto :err
echo == 4. 搜索树可视化 成功 / 启发误导 / LLM 误杀 ==
python -m experiments.run_visuals || goto :err
echo == 5. 失败案例挖掘 + h 适应度地形扫描 ==
python -m experiments.run_failure_analysis || goto :err
python -m experiments.run_hparam_sweep || goto :err
python -m experiments.run_decision_eval || goto :err
echo == 6. 内核单元测试 ==
python -m tests.test_kernels || goto :err
echo == 7. 报告数字与产物一致性 ==
python -m tools.check_report_numbers || goto :err
echo 完成。查看 results/main_table.md / results/figs/ 与 results/trees/
goto :eof
:err
echo 执行失败，请检查上方输出。
exit /b 1