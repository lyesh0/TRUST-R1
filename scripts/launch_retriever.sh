#!/usr/bin/env bash
# =============================================================================
# TRUST-R1 AutoDL Retriever 启动脚本 (前台运行，适合调试)
# =============================================================================
# 用法: bash scripts/launch_retriever.sh
# 后台运行请用: bash retrieval_launch.sh
# 所有路径参考: docs/autodl_paths.md
# =============================================================================
set -euo pipefail

# Path defaults (override via env vars)
SEARCH_DATA_ROOT="${SEARCH_DATA_ROOT:-/root/autodl-fs}"
CORPUS_EXTRACTED="${CORPUS_EXTRACTED:-$SEARCH_DATA_ROOT/data/wiki-18-extracted.jsonl}"
INDEX_TEST="${INDEX_TEST:-$SEARCH_DATA_ROOT/test-index-100K.faiss}"
INDEX_FULL="${INDEX_FULL:-$SEARCH_DATA_ROOT/indexes/wiki-18/e5_Flat.index}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-/root/e5-base-v2}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/root/miniconda3/envs/retriever/bin/python}"

TOPK="${TOPK:-3}"
RETRIEVER_MODE="${RETRIEVER_MODE:-test}"

# Disable Intel ITT profiling
export INTEL_LIBITTNOTIFY64=
export INTEL_JIT_NOTIFIER_DISABLE=1

# Select mode
case "$RETRIEVER_MODE" in
    test)
        SERVER_SCRIPT="search_r1/search/retrieval_server.py"
        ARGS=(
            --index_path "$INDEX_TEST"
            --corpus_path "$CORPUS_EXTRACTED"
            --topk "$TOPK"
            --retriever_name e5
            --retriever_model "$RETRIEVER_MODEL"
        )
        ;;
    bm25)
        SERVER_SCRIPT="search_r1/search/bm25_fast_server.py"
        ARGS=(--corpus_path "$CORPUS_EXTRACTED" --topk "$TOPK")
        ;;
    bm25-stream)
        SERVER_SCRIPT="search_r1/search/simple_bm25_server.py"
        ARGS=(--corpus_path "$CORPUS_EXTRACTED" --topk "$TOPK")
        ;;
    *)
        echo "ERROR: Unknown mode: $RETRIEVER_MODE" >&2
        exit 1
        ;;
esac

echo "=== Launching Retriever (mode=$RETRIEVER_MODE) ==="
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec "$RETRIEVER_PYTHON" $SERVER_SCRIPT "${ARGS[@]}"
