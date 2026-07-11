# AutoDL 工作流

本文件记录 TRUST-R1 在 AutoDL 上进行环境、数据、训练和评测的操作规范。

## 1. 基本原则

- GitHub 是代码唯一主版本。
- 本地 Mac 只做轻量开发和文档工作。
- AutoDL 负责数据、模型、索引、retriever、训练、评测和 checkpoint。
- 正式实验必须对应 git commit hash。
- 不要把数据、模型、索引、checkpoint、大日志提交到 git。

## 2. 推荐路径

根据 AutoDL 实际挂载情况调整，最终路径需要记录到 `reports/environment_lock.md`。

```text
/root/autodl-tmp/TRUST-R1/       # code repo
/root/autodl-tmp/data/nq_search/ # small train/test parquet used by RL
/root/autodl-tmp/models/         # model weights
/root/autodl-tmp/runs/           # training outputs/checkpoints/logs
/root/autodl-tmp/reports/        # small exported summaries
/root/autodl-fs/data/            # shared search corpus, e.g. wiki-18.jsonl
/root/autodl-fs/indexes/         # shared FAISS/BM25 indexes
```

4×4090 smoke test 默认使用 `/root/autodl-fs` 存放搜索语料和 index；训练输出仍写入 `/root/autodl-tmp/runs`，避免把 checkpoint/log 放进共享搜索数据目录。

如需准备 wiki-18 检索语料和 e5 index，在 AutoDL 上运行：

```bash
cd /root/autodl-tmp/TRUST-R1
SEARCH_DATA_ROOT=/root/autodl-fs
python scripts/download.py --data-root "$SEARCH_DATA_ROOT"
```

该命令会生成 `/root/autodl-fs/data/wiki-18.jsonl` 和 `/root/autodl-fs/indexes/wiki-18/e5_Flat.index`。

## 3. 代码同步

本地开发后：

```bash
git status
git add ...
git commit -m "..."
git push
```

AutoDL 开始工作前：

```bash
git pull
```

AutoDL 远程 debug 后：

```bash
git status
git add ...
git commit -m "..."
git push
```

## 4. 训练前 checklist

```text
[ ] 当前 shell 在 AutoDL，不在本地 Mac
[ ] git status 干净，或 uncommitted diff 已明确记录
[ ] git commit hash 已记录
[ ] config 文件路径正确
[ ] 数据路径存在
[ ] 模型路径存在
[ ] 4 卡训练配置一致：`trainer.n_gpus_per_node=4`，`CUDA_VISIBLE_DEVICES=0,1,2,3`
[ ] 搜索语料和 index 位于 `/root/autodl-fs`，例如 `/root/autodl-fs/data/wiki-18.jsonl` 和 `/root/autodl-fs/indexes/wiki-18/e5_Flat.index`
[ ] index / retriever 可用；4×4090 smoke test 默认用 CPU retriever（`--faiss-gpu false`），避免占用训练 GPU
[ ] fault seed / train seed 已写入配置
[ ] 已跑过 2-step dry run
[ ] 输出目录是新的 run_id
[ ] checkpoint / log 不会进 git
```

## 5. Run ID 命名

推荐格式：

```text
YYYYMMDD_<experiment_id>_<setting>_seed<seed>
```

示例：

```text
20260702_B0_clean_seed42
20260702_B1_faultaug20_seed42
20260702_M1_recovery_seed42
20260702_M2_full_seed42
```

## 6. 每次实验至少保存

```text
run_id
commit hash
config 文件
启动命令
开始/结束时间
GPU 类型和数量
数据 split
模型 checkpoint 路径
fault seed
train seed
训练日志路径
评测输出路径
抽样 trajectory 路径
```

## 7. 结果同步

可以同步回 git 的小文件：

- `metrics.json`
- `summary.md`
- 小型抽样 `selected_trajectories.jsonl`
- 图表
- 环境记录

不要同步回 git：

- checkpoint
- 全量 trajectory dump
- 模型权重
- 数据集
- index
- 大型日志

## 8. 止损规则

- 环境一天内无法跑通：回退到原 Search-R1 支持良好的路线。
- 全量 index 卡住：先用小语料或小索引跑通 pipeline。
- 7B Base 格式长期不稳定：先 format reward，再 format-only SFT，最后再考虑 Instruct。
- M1/M2 没提升：先检查 fault 是否触发、reward 是否稀疏、query novelty 阈值是否合理，不要继续堆复杂模块。
