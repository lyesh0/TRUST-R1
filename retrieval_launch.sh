#!/usr/bin/env bash
set -euo pipefail

SEARCH_DATA_ROOT="${SEARCH_DATA_ROOT:-/root/autodl-fs}"
INDEX_FILE="${INDEX_FILE:-$SEARCH_DATA_ROOT/indexes/wiki-18/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-$SEARCH_DATA_ROOT/data/wiki-18-extracted.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-/root/e5-base-v2}"
TOPK="${TOPK:-3}"
FAISS_GPU="${FAISS_GPU:-false}"

faiss_flag=()
if [[ "$FAISS_GPU" == "true" || "$FAISS_GPU" == "1" || "$FAISS_GPU" == "yes" ]]; then
  faiss_flag=(--faiss_gpu)
fi

python search_r1/search/retrieval_server.py \
  --index_path "$INDEX_FILE" \
  --corpus_path "$CORPUS_FILE" \
  --topk "$TOPK" \
  --retriever_name "$RETRIEVER_NAME" \
  --retriever_model "$RETRIEVER_MODEL" \
  "${faiss_flag[@]}"
