@echo off
chcp 65001 >nul
cd /d %~dp0
set PYTHONIOENCODING=utf-8
echo == 0. 依赖自检 ==
python -c "import numpy, scipy, sklearn, pandas, matplotlib, yaml, lightgbm" || goto :err
echo == 1. 分层切分 dev / heldout / train_rest ==
python -m data.ace_loader || goto :err
echo == 2. 全量条款图 + 覆盖率 + 20 条边抽查 ==
python -m data.clause_graph_builder || goto :err
python -m tools.audit_clause_graph || goto :err
echo == 3. 真实轨迹训练路径价值分类器（LightGBM） ==
python -m experiments.train_lgb_classifier --out models\pc_new.lgb || goto :err
move /y models\pc_new.lgb models\path_classifier.lgb >nul || goto :err
echo == 4. GA 优化 h_theta，dev 适应度 + held-out 监控 ==
python -m experiments.run_ga || goto :err
echo == 5. dev 上 λ 融合权重小网格（18 组） ==
python -m experiments.run_lambda_grid --limit 300 || goto :err
echo == 6. 主表 8 条件 x dev，随后 test 只跑一次 ==
python -m experiments.run_main_table --allow-test || goto :err
echo == 7. 搜索树可视化 成功 / 启发误导 / LLM 误杀 ==
python -m experiments.run_visuals || goto :err
echo == 8. 失败案例挖掘 + h 适应度地形扫描 + 判据评估 + 按跳数分层 ==
python -m experiments.run_failure_analysis || goto :err
python -m experiments.run_hparam_sweep || goto :err
python -m experiments.run_decision_eval || goto :err
python -m experiments.run_stratified || goto :err
python -m experiments.run_hallucination_sweep || goto :err
echo == 9. LLM 误杀率 + 调用成本表 ==
python -m tools.llm_reject_report || goto :err
python -m tools.llm_cost_table || goto :err
echo == 10. 内核单元测试 ==
python -m tests.test_kernels || goto :err
echo == 11. 报告数字与产物一致性 ==
python -m tools.check_report_numbers || goto :err
echo 完成。查看 results/main_table.md / results/figs/ 与 results/trees/
goto :eof
:err
echo 执行失败，请检查上方输出。
exit /b 1
