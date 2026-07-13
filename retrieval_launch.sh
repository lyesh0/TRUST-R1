#!/usr/bin/env bash
# =============================================================================
# TRUST-R1 Retriever Launch Script (AutoDL)
# =============================================================================
#
# 此脚本启动本地检索服务器，支持多种检索模式。
# 所有重要路径集中在此文件和 docs/autodl_paths.md 中。
#
# 用法:
#   # 默认: E5 Dense 全量 (需要解决的 FAISS mmap 内存问题)
#   bash retrieval_launch.sh
#
#   # BM25 快速版 (内存倒排索引)
#   RETRIEVER_MODE=bm25 bash retrieval_launch.sh
#
#   # E5 Dense 测试版 (100K 索引，已验证可用)
#   RETRIEVER_MODE=test bash retrieval_launch.sh
#
#   # BM25 流式版 (小内存，速度较慢)
#   RETRIEVER_MODE=bm25-stream bash retrieval_launch.sh
#
# =============================================================================
set -euo pipefail

# =============================================================================
# 路径常量 (所有数据文件集中于此)
# =============================================================================

# 数据根目录
SEARCH_DATA_ROOT="${SEARCH_DATA_ROOT:-/root/autodl-fs}"

# --- 语料文件 ---
# 原始 tar 包 (包含 wiki_dump.jsonl):       /root/autodl-fs/data/wiki-18.jsonl
# 解压后的纯 JSONL (推荐使用):             /root/autodl-fs/data/wiki-18-extracted.jsonl
CORPUS_TAR="${CORPUS_TAR:-$SEARCH_DATA_ROOT/data/wiki-18.jsonl}"
CORPUS_EXTRACTED="${CORPUS_EXTRACTED:-$SEARCH_DATA_ROOT/data/wiki-18-extracted.jsonl}"
CORPUS_FILE="${CORPUS_FILE:-$CORPUS_EXTRACTED}"

# --- FAISS 索引 ---
# 全量 E5 Flat 索引 (61GB，需解决 mmap 内存问题): /root/autodl-fs/indexes/wiki-18/e5_Flat.index
# 测试索引 100K (293MB，已验证可用):              /root/autodl-fs/test-index-100K.faiss
INDEX_FULL="${INDEX_FULL:-$SEARCH_DATA_ROOT/indexes/wiki-18/e5_Flat.index}"
INDEX_TEST="${INDEX_TEST:-$SEARCH_DATA_ROOT/test-index-100K.faiss}"
INDEX_FILE="${INDEX_FILE:-$INDEX_FULL}"

# --- 检索模型 ---
# E5-base-v2 (已下载):  /root/e5-base-v2/
# BGE-small-en (备用):   /root/autodl-tmp/models/models/AI-ModelScope--bge-small-en-v1.5
RETRIEVER_MODEL_DIR="${RETRIEVER_MODEL_DIR:-/root/e5-base-v2}"

# --- Conda 环境 ---
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-/root/miniconda3/envs/retriever}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-$RETRIEVER_CONDA_ENV/bin/python}"

# --- 日志与 PID ---
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/TRUST-R1}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$SEARCH_DATA_ROOT/retriever.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/retriever.pid}"

# --- 检索参数 ---
TOPK="${TOPK:-3}"
RETRIEVER_MODE="${RETRIEVER_MODE:-test}"  # test | bm25 | bm25-stream | dense-full
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# =============================================================================
# 前置检查
# =============================================================================

# Disable Intel ITT profiling (fix MKL 2025 symbol conflict)
export INTEL_LIBITTNOTIFY64=
export INTEL_JIT_NOTIFIER_DISABLE=1

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# 根据模式选择文件
case "$RETRIEVER_MODE" in
    test)
        SERVER_SCRIPT="search_r1/search/retrieval_server.py"
        INDEX_FILE="$INDEX_TEST"
        CORPUS_FILE="${CORPUS_TEST:-$SEARCH_DATA_ROOT/test-corpus-100K.jsonl}"
        RETRIEVER_NAME="e5"
        ;;
    bm25)
        SERVER_SCRIPT="search_r1/search/bm25_fast_server.py"
        CORPUS_FILE="$CORPUS_EXTRACTED"
        RETRIEVER_NAME="bm25"
        ;;
    bm25-stream)
        SERVER_SCRIPT="search_r1/search/simple_bm25_server.py"
        CORPUS_FILE="$CORPUS_EXTRACTED"  # 已解压版本无需跳过 tar header
        RETRIEVER_NAME="bm25-stream"
        ;;
    dense-full)
        SERVER_SCRIPT="search_r1/search/retrieval_server.py"
        INDEX_FILE="$INDEX_FULL"
        CORPUS_FILE="$CORPUS_EXTRACTED"
        RETRIEVER_NAME="e5"
        ;;
    *)
        echo "ERROR: Unknown RETRIEVER_MODE: $RETRIEVER_MODE" >&2
        echo "  Valid modes: test, bm25, bm25-stream, dense-full" >&2
        exit 1
        ;;
esac

echo "=== TRUST-R1 Retriever ==="
echo "  Mode:       $RETRIEVER_MODE"
echo "  Script:     $SERVER_SCRIPT"
echo "  Corpus:     $CORPUS_FILE"
echo "  Index:      ${INDEX_FILE:-N/A}"
echo "  Model:      ${RETRIEVER_MODEL_DIR:-N/A}"
echo "  TopK:       $TOPK"
echo "  Host:Port:  $HOST:$PORT"
echo "  Log:        $LOG_FILE"
echo ""

# 检查文件是否存在
if [[ ! -f "$CORPUS_FILE" ]]; then
    echo "ERROR: Corpus not found: $CORPUS_FILE" >&2
    echo "  Hint: tar xf /root/autodl-fs/data/wiki-18.jsonl -O > /root/autodl-fs/data/wiki-18-extracted.jsonl" >&2
    exit 1
fi

# 构建命令行参数
ARGS=()

case "$RETRIEVER_MODE" in
    test|dense-full)
        if [[ ! -f "$INDEX_FILE" ]]; then
            echo "ERROR: Index not found: $INDEX_FILE" >&2
            exit 1
        fi
        if [[ ! -d "$RETRIEVER_MODEL_DIR" ]]; then
            echo "ERROR: Model not found: $RETRIEVER_MODEL_DIR" >&2
            exit 1
        fi
        ARGS=(
            --index_path "$INDEX_FILE"
            --corpus_path "$CORPUS_FILE"
            --topk "$TOPK"
            --retriever_name "$RETRIEVER_NAME"
            --retriever_model "$RETRIEVER_MODEL_DIR"
        )
        ;;
    bm25|bm25-stream)
        ARGS=(
            --corpus_path "$CORPUS_FILE"
            --topk "$TOPK"
            --host "$HOST"
            --port "$PORT"
        )
        ;;
esac

# 停掉旧进程
OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
if [[ -n "$OLD_PID" ]] && ps -p "$OLD_PID" > /dev/null 2>&1; then
    echo "Stopping old retriever (PID: $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 2
fi
fuser -k "${PORT}/tcp" 2>/dev/null || true

# 启动
cd "$PROJECT_ROOT"
nohup "$RETRIEVER_PYTHON" $SERVER_SCRIPT "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
RETRIEVER_PID=$!
echo "$RETRIEVER_PID" > "$PID_FILE"

echo "Started with PID: $RETRIEVER_PID"
echo "Waiting for server ready (may take up to 60s)..."
echo "  tail -f $LOG_FILE"

# 等待服务器就绪
for i in $(seq 1 60); do
    sleep 2
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "Retriever is READY! PID: $RETRIEVER_PID"
        echo "API: http://127.0.0.1:${PORT}/retrieve"
        echo "Health: http://127.0.0.1:${PORT}/health"
        exit 0
    fi
    if ! ps -p "$RETRIEVER_PID" > /dev/null 2>&1; then
        echo "ERROR: Retriever process died during startup!"
        echo "=== Last 30 lines of log ==="
        tail -30 "$LOG_FILE"
        exit 1
    fi
done

echo "NOTE: Retriever still initializing after 120s, check log: tail -f $LOG_FILE"
