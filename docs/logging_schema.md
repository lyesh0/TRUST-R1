# 日志与轨迹 Schema

本文件定义 TRUST-R1 的日志、轨迹和指标记录约定。首版实现可以从最小字段开始，但正式实验必须保证可追踪、可复现。

## 1. Run 目录结构

推荐每个正式实验保存为：

```text
runs/<run_id>/
├── config.yaml
├── command.sh
├── environment.md
├── metrics.json
├── summary.md
├── trajectories.jsonl          # AutoDL 保存，通常不进 git
└── selected_trajectories.jsonl # 小型抽样，可进 git
```

## 2. Run 级元信息

```json
{
  "run_id": "20260702_M2_full_seed42",
  "experiment_id": "M2",
  "git_commit": "...",
  "host": "autodl",
  "gpu": "4xA800-80GB",
  "model": "Qwen/Qwen2.5-7B",
  "dataset": "NQ+HotpotQA",
  "retriever": "E5+FAISS",
  "fault_setting": "mixed20",
  "train_seed": 42,
  "fault_seed": 42
}
```

## 3. Trajectory JSONL 单行建议格式

```json
{
  "run_id": "20260702_M2_full_seed42",
  "sample_id": "nq_dev_000001",
  "question": "...",
  "gold_answer": "...",
  "final_answer": "...",
  "is_correct": true,
  "turns": [
    {
      "step": 1,
      "action": "search",
      "query": "...",
      "fault_enabled": true,
      "fault_type": "drop_top",
      "retrieved_doc_ids": ["doc1", "doc2", "doc3"],
      "observation_chars": 1200
    }
  ],
  "recovery": {
    "had_fault": true,
    "searched_again_after_fault": true,
    "changed_query_after_fault": true,
    "answered_correctly_after_fault": true
  },
  "reward": {
    "answer": 1.0,
    "format": 0.0,
    "recovery": 0.2,
    "duplicate_penalty": 0.0,
    "total": 1.2
  }
}
```

## 4. Fault Event 字段

每次检索故障至少记录：

- `fault_enabled`；
- `fault_type`；
- `fault_rate`；
- `fault_seed`；
- 原始 top-k doc ids；
- 故障后 doc ids；
- 是否为空返回；
- 是否重复返回；
- 是否 drop top evidence。

## 5. Metrics JSON 建议字段

```json
{
  "exact_match": 0.0,
  "f1": 0.0,
  "clean_accuracy": 0.0,
  "noisy_accuracy": 0.0,
  "robustness_drop": 0.0,
  "first_failure_recovery_rate": 0.0,
  "average_search_calls": 0.0,
  "query_rewrite_rate": 0.0,
  "duplicate_query_rate": 0.0,
  "invalid_action_rate": 0.0
}
```

所有字段初始可以为 0 或缺省，但报告中使用的指标必须能追溯到真实计算脚本和日志。

## 6. Git 规则

可以进入 git：

- 小型 `metrics.json`；
- `summary.md`；
- 抽样 `selected_trajectories.jsonl`；
- 图表；
- schema 文档。

不要进入 git：

- 全量 trajectory dump；
- checkpoint；
- 模型权重；
- 数据集；
- index；
- 大型日志。
