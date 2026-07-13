"""
Simple BM25 retrieval server with on-demand search (no pre-built index).
"""
import json
import re
from typing import List, Dict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import argparse


class SimpleBM25Retriever:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.header_offset = 0
        # Auto-detect tar header (starts with file path, not '{')
        with open(self.corpus_path, 'rb') as f:
            peek = f.read(512)
            if not peek.startswith(b'{'):
                self.header_offset = 512
        print(f"Corpus ready: {corpus_path} (header_offset={self.header_offset})")

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for top-k documents using simple term matching."""
        query_tokens = self._tokenize(query)
        results = []

        # Read file content
        with open(self.corpus_path, 'rb') as f:
            # Skip header if present (auto-detected)
            if self.header_offset:
                f.read(self.header_offset)
            content = f.read()

        # Process documents and score
        for line_bytes in content.split(b'\n'):
            if not line_bytes:
                continue
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line.startswith('{'):
                continue
            try:
                doc = json.loads(line)
                text = doc.get('contents', '').lower()
                tokens = set(self._tokenize(text))

                # Count matching terms
                match_count = sum(1 for term in query_tokens if term in tokens)

                if match_count > 0:
                    results.append({'contents': doc.get('contents', ''), '_score': match_count})
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort and return top k
        results.sort(key=lambda x: x['_score'], reverse=True)
        return results[:k]


class QueryRequest(BaseModel):
    queries: List[str]
    topk: int = 3


app = FastAPI()
retriever = None


def init_retriever(corpus_path):
    global retriever
    print(f"Loading retriever from {corpus_path}...")
    retriever = SimpleBM25Retriever(corpus_path)
    print("Retriever loaded!")
    return retriever


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """Retrieval endpoint matching the expected API."""
    results = []
    for query in request.queries:
        docs = retriever.search(query, k=request.topk)
        results.append(docs)
    return {"result": results}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "bm25-retriever", "ready": retriever is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch simple BM25 retriever.")
    parser.add_argument("--corpus_path", type=str, default="/root/autodl-fs/data/wiki-18.jsonl")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Initialize retriever before starting server
    print(f"Initializing BM25 retriever...")
    print(f"  Corpus: {args.corpus_path}")
    init_retriever(args.corpus_path)

    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)