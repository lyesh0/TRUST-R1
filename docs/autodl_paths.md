# AutoDL 路径参考

> 最后更新: 2026-07-12
> 所有路径基于 AutoDL RTX 4090 (120GB RAM, 24GB GPU) 实例。

## 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | RTX 4090 (24GB) × 1 |
| CPU | 16 vCPU Intel Xeon Gold 6430 |
| 内存 | 120GB (cgroup 限制: 128849018880 bytes) |
| 系统盘 | 30GB (`/`) |
| 数据盘 | 80GB (`/root/autodl-tmp`) + 200GB (`/autodl-fs`) |

## 项目路径

```
/root/autodl-tmp/TRUST-R1/          # 项目根目录
  retrieval_launch.sh               # Retriever 启动 (后台, 多模式)
  scripts/launch_retriever.sh       # Retriever 启动 (前台, 调试用)
  search_r1/search/                  # Retriever 源代码
    retrieval_server.py              # Dense retriever (FAISS + e5/bge)
    bm25_fast_server.py              # Fast BM25 (内存倒排索引)
    simple_bm25_server.py            # Simple BM25 (流式, 小内存)
    index_builder.py                 # FAISS/BM25 索引构建
    build_index.sh                   # 索引构建命令示例
  docs/
    autodl_paths.md                  # 本文件
    autodl_workflow.md               # AutoDL 工作流
```

## 数据文件

### 语料 (Corpus)

| 路径 | 大小 | 说明 |
|------|------|------|
| `/root/autodl-fs/data/wiki-18.jsonl` | 14GB | **原始 tar 包**，内含 `wiki_dump.jsonl` |
| `/root/autodl-fs/data/wiki-18-extracted.jsonl` | 14GB | **解压后的纯 JSONL** (21M 行)，推荐使用 |

```bash
# 如果还没解压:
tar xf /root/autodl-fs/data/wiki-18.jsonl -O > /root/autodl-fs/data/wiki-18-extracted.jsonl
```

### FAISS 索引

| 路径 | 大小 | 向量数 | 维度 | 说明 |
|------|------|--------|------|------|
| `/root/autodl-fs/indexes/wiki-18/e5_Flat.index` | 61GB | ~21M | 768 | **全量 E5 Flat IP 索引** — 120GB 内存下 `read_index` 会 OOM |
| `/root/autodl-fs/test-index-100K.faiss` | 293MB | 100K | 768 | **测试索引** — 已验证可用 |

> **已知问题**: 全量 61GB 索引在 AutoDL 120GB 内存限制下，`faiss.read_index` 即使加了 `IO_FLAG_MMAP` 也会将整个文件拷贝到物理内存导致 OOM。
> **临时方案**: 使用 100K 测试索引进行开发，或安装 Java 后构建 PySerini BM25 Lucene 索引。

### 测试文件

| 路径 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `/root/autodl-fs/test-corpus-100K.jsonl` | 65MB | 100K | 测试语料 (wiki-18 前 10 万行) |
| `/root/autodl-fs/wiki-18-small.jsonl` | 663KB | 1K | 微型测试语料 |
| `/root/autodl-fs/wiki-18-bm25-test-clean.jsonl` | 6.5MB | 10K | BM25 测试语料 |

### 其他测试数据

```
/root/autodl-fs/test-corpus.jsonl           # 9 条示例数据
/root/autodl-fs/wiki-18-bm25-test.jsonl     # BM25 原始测试数据
/root/autodl-tmp/corpus/corpus.jsonl        # 旧路径 (6.3KB 样本)
/root/autodl-tmp/corpus/index/bge_Flat.index # 旧路径 (16KB 桩)
```

## 模型文件

| 路径 | 大小 | 说明 |
|------|------|------|
| `/root/e5-base-v2/` | 419MB | **E5-base-v2** — 与 e5_Flat.index 匹配 |
| `/root/autodl-tmp/models/Qwen2.5-3B-Instruct/` | ~6GB | 训练用 LLM |
| `/root/autodl-tmp/models/models/AI-ModelScope--bge-small-en-v1.5/` | - | BGE-small (备用) |

HF 缓存: `/root/autodl-tmp/cache/huggingface/hub/`

## Conda 环境

| 环境名 | Python | 路径 | 用途 |
|------|------|------|------|
| `retriever` | 3.10 | `/root/miniconda3/envs/retriever` | Retriever 服务 |
| `trustr1` | 3.10 | `/root/miniconda3/envs/trustr1` | RL 训练 |
| `base` | 3.12 | `/root/miniconda3` | 系统默认 |

Retriever 环境关键包: `faiss-gpu=1.14.3`, `torch=2.4.0`, `transformers`, `fastapi`, `uvicorn`, `pyserini`, `datasets`

## 日志文件

```
/root/autodl-fs/retriever.log                # Retriever 主日志
/root/autodl-fs/bm25_retriever.log           # BM25 旧日志
/root/autodl-tmp/TRUST-R1/logs/retriever.pid # Retriever 进程 PID
```

## Retriever 启动快速参考

```bash
# 测试模式 (E5 Dense + 100K 索引，推荐开发用)
RETRIEVER_MODE=test bash retrieval_launch.sh

# BM25 快速模式 (内存倒排索引)
RETRIEVER_MODE=bm25 bash retrieval_launch.sh

# BM25 流式模式 (极低内存)
RETRIEVER_MODE=bm25-stream bash retrieval_launch.sh

# 前台调试
bash scripts/launch_retriever.sh
```

## API 端点

```
GET  http://127.0.0.1:8000/health           # 健康检查
POST http://127.0.0.1:8000/retrieve          # 检索
  Body: {"queries": ["query1", "query2"], "topk": 3}
  Response: {"result": [[{doc}, ...], [...]]}
```

## 端口

| 端口 | 服务 |
|------|------|
| 8000 | Retriever API |
| 6006 | TensorBoard (开放) |
| 6008 | 自定义 HTTP (开放) |
