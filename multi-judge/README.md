# Multi-Judge v6

当前冻结的三模型四路线评分方案。默认 runner 使用 v6 rubric，并保留每个模型、每轮、每条路线的分数、证据、理由、自报置信度和重试记录。模型间一致性（Fleiss/Cohen κ）不代表相对人工金标准的准确率；confidence 是模型自报值，未做概率校准。

## 运行

依赖：`pip install -r multi-judge/requirements.txt`。设置兼容 OpenAI Chat Completions 的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`，并按实验配置设置 `EM_JUDGE_REASONING=1`。

    python multi-judge/run_multi_judge.py experiments/data/gptoss5000_induced.json \
      --output-dir /path/to/run --benchmark travelbench --sample-size 100 \
      --seed 20261102 --threshold 60 --max-completion-tokens 8000 \
      --concurrency 8 --task-concurrency 6 --max-retries 8

默认模型为 `deepseek/deepseek-v4.1-flash-1017`、`z-ai/glm-5.3-flash`、`xiaomi/mimo-v2.6-pro`。续跑时必须使用同一 manifest 配置；不能覆盖已完成的不同实验。v6 rubrics、冻结源码、哈希及开发/留出结果保存在 [`archive/v6`](archive/v6/README.md)。

输出包含固定样本、逐路线原始 judgments、analysis 和失败记录。`export_report.py` 导出逐轨迹 CSV；`compare_runs.py` 比较两轮冻结运行；`prepare_holdout.py` 和 `report_v6.py` 用于任务隔离及留出集评估；`report_holdout.py` 汇总留出统计。`review_results.py`、`audit_disagreements.py` 生成离线复核材料，不改写源结果。

单模型四路评分仍可用 `eval_em_multi.py`。其 prompt 与输入/输出接口独立于三模型交叉验证流程。

## 验证

    python -m unittest discover -s multi-judge/tests -v

测试为离线测试，不调用评分 API。冻结 v6 开发集 κ=0.8520；新任务留出集 κ=0.8044（95% CI [0.7029, 0.8916]），留出集存在 1 个无效评分，因此完整覆盖预设未通过。详见归档报告。统计衡量标注者一致性，不是准确率。
