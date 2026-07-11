#!/usr/bin/env bash
# TRUST-R1 AutoDL Retriever 启动脚本
# 参考 README-searchr1.md 的 retriever 环境配置

set -euo pipefail

# 要求在 retriever 环境中运行
if [[ "${CONDA_DEFAULT_ENV:-}" != "retriever" ]]; then
  echo "错误: 请在 retriever conda 环境中运行" >&2
  echo "  conda activate retriever" >&2
  echo "  bash scripts/launch_retriever.sh" >&2
  exit 1
fi

# AutoDL 路径配置
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"

# 索引与语料文件
SEARCH_DATA_ROOT="${SEARCH_DATA_ROOT:-$AUTODL_ROOT/corpus}"
INDEX_FILE="${INDEX_FILE:-$SEARCH_DATA_ROOT/index/bge_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-$SEARCH_DATA_ROOT/corpus.jsonl}"

# Retriever 配置
RETRIEVER_NAME="${RETRIEVER_NAME:-bge}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$AUTODL_ROOT/models/bge-small-en-v1.5}"
TOPK="${TOPK:-3}"

# FAISS GPU 配置 (AutoDL 4090 可选)
FAISS_GPU="${FAISS_GPU:-false}"

faiss_flag=()
if [[ "$FAISS_GPU" == "true" || "$FAISS_GPU" == "1" || "$FAISS_GPU" == "yes" ]]; then
  faiss_flag=(--faiss_gpu)
fi

# 检查文件是否存在
if [[ ! -f "$CORPUS_FILE" ]]; then
  echo "错误: 语料不存在: $CORPUS_FILE" >&2
  exit 1
fi

if [[ ! -f "$INDEX_FILE" ]]; then
  echo "错误: 索引不存在: $INDEX_FILE" >&2
  exit 1
fi

echo "=== 启动 Retriever ==="
echo "  索引: $INDEX_FILE"
echo "  语料: $CORPUS_FILE"
echo "  Retriever: $RETRIEVER_NAME"
echo "  模型: $RETRIEVER_MODEL"
echo "  TopK: $TOPK"
echo "  FAISS GPU: $FAISS_GPU"
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python search_r1/search/retrieval_server.py \
  --index_path "$INDEX_FILE" \
  --corpus_path "$CORPUS_FILE" \
  --topk "$TOPK" \
  --retriever_name "$RETRIEVER_NAME" \
  --retriever_model "$RETRIEVER_MODEL" \
  "${faiss_flag[@]}"