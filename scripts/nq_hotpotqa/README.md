
## Reproduce the paper results

### Download the dataset

```bash
huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train --local-dir $WORK_DIR/data/nq_hotpotqa_train
```

### Launch the local search engine

(1) Download and prepare the indexing and corpus. On AutoDL, keep shared retriever assets under `/root/autodl-fs`.
```bash
SEARCH_DATA_ROOT=/root/autodl-fs
python scripts/download.py --data-root "$SEARCH_DATA_ROOT"
```

This creates `$SEARCH_DATA_ROOT/indexes/wiki-18/e5_Flat.index` and `$SEARCH_DATA_ROOT/data/wiki-18.jsonl`.

(2) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

### Run PPO training
```bash
bash train_ppo.sh
```


### Run GRPO training
```bash
bash train_grpo.sh
```

### Run evaluation
```bash
bash evaluate.sh
```

You can change ```$BASE_MODEL``` to the path of the model you would like to evaluate.
