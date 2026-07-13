"""
Fast BM25 retrieval server with efficient indexing.
"""
import json
import re
from typing import List, Dict, Set
from collections import defaultdict
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import argparse


class FastBM25Retriever:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        print(f"Building inverted index from {corpus_path}...")
        # Build inverted index
        self.inverted_index = defaultdict(list)  # term -> list of doc positions
        self.doc_positions = []  # position -> (byte_start, byte_end)
        self._build_index()
        print(f"Index built: {len(self.inverted_index)} terms, {len(self.doc_positions)} documents")

    def _build_index(self):
        pos = 512  # Skip binary header

        with open(self.corpus_path, 'rb') as f:
            while True:
                byte_start = f.tell()
                line_bytes = f.readline()
                if not line_bytes:
                    break

                byte_end = byte_start + len(line_bytes)
                line = line_bytes.decode('utf-8', errors='ignore').strip()

                if line.startswith('{'):
                    try:
                        doc = json.loads(line)
                        text = doc.get('contents', '').lower()

                        # Tokenize and build inverted index
                        tokens = re.findall(r'[a-z0-9]+', text)
                        seen = set()  # Deduplicate terms per document
                        for token in tokens:
                            if token not in seen:
                                self.inverted_index[token].append(len(self.doc_positions))
                                seen.add(token)

                        self.doc_positions.append((byte_start, byte_end))
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r'[a-z0-9]+', text)

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search using inverted index."""
        query_tokens = set(self._tokenize(query))

        if not query_tokens:
            return []

        # Find documents containing query terms
        candidate_docs = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])

        # Score candidates
        scored = {}
        for doc_idx in candidate_docs:
            # Get document text
            byte_start, byte_end = self.doc_positions[doc_idx]
            with open(self.corpus_path, 'rb') as f:
                f.seek(byte_start)
                line_bytes = f.read(byte_end - byte_start)
                doc = json.loads(line_bytes.decode('utf-8', errors='ignore'))
                text = doc.get('contents', '').lower()

                # Score by term matches
                doc_tokens = set(re.findall(r'[a-z0-9]+', text))
                matches = len(query_tokens & doc_tokens)

                if matches > 0:
                    scored[doc_idx] = (matches, doc.get('contents', ''))

        # Sort by match count and return top k
        sorted_results = sorted(scored.items(), key=lambda x: x[1][0], reverse=True)

        return [{'contents': item[1]} for _, item in sorted_results[:k]]


class QueryRequest(BaseModel):
    queries: List[str]
    topk: int = 3


app = FastAPI()
retriever = None


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
    return {"status": "ok", "service": "bm25-fast", "ready": retriever is not None}


if __name__ == "__main__":
    def init_retriever(corpus_path):
        global retriever
        print(f"Loading retriever from {corpus_path}...")
        retriever = FastBM25Retriever(corpus_path)
        print("Retriever loaded!")
        return retriever

    parser = argparse.ArgumentParser(description="Launch fast BM25 retriever.")
    parser.add_argument("--corpus_path", type=str, default="/root/autodl-fs/data/wiki-18.jsonl")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Initialize retriever
    print(f"Initializing BM25 retriever...")
    init_retriever(args.corpus_path)

    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)