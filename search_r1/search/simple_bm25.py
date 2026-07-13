"""
Simple BM25 retriever implementation without external dependencies.
"""
import math
import re
from typing import List, Dict, Tuple
import json
from collections import Counter


class SimpleBM25Retriever:
    def __init__(self, corpus_path: str, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            corpus_path: Path to the corpus JSONL file
            k1: Term saturation parameter
            b: Length normalization parameter
        """
        self.corpus_path = corpus_path
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}  # term -> (doc_count, collection_freq)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.N = 0  # number of documents
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens

    def _build_index(self):
        """Build the BM25 index."""
        print("Building BM25 index...")
        doc_lengths = []
        term_doc_count = Counter()
        term_collection_freq = Counter()
        N = 0

        with open(self.corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                text = doc.get('contents', '')
                tokens = self._tokenize(text)
                doc_lengths.append(len(tokens))

                # Count terms in this document
                term_counts = Counter(tokens)
                for term, count in term_counts.items():
                    term_doc_count[term] += 1
                    term_collection_freq[term] += count

                N += 1
                if N % 100000 == 0:
                    print(f"  Processed {N} documents...")

        self.N = N
        self.doc_lengths = doc_lengths
        self.avg_doc_length = sum(doc_lengths) / N if N > 0 else 0
        self.doc_freqs = {
            term: (term_doc_count[term], term_collection_freq[term])
            for term in term_doc_count
        }

        print(f"Index built: {N} documents, {len(self.doc_freqs)} unique terms")

    def _score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Calculate BM25 score for a query-document pair."""
        doc_token_counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)

        score = 0
        for term in query_tokens:
            if term in doc_token_counts:
                # Term frequency in document
                tf = doc_token_counts[term]
                # Document frequency and collection frequency
                df, _ = self.doc_freqs.get(term, (0, 0))
                # IDF
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator

        return score

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        Search for top-k documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (doc_id, score, doc_dict) tuples
        """
        query_tokens = self._tokenize(query)
        results = []

        with open(self.corpus_path, 'r') as f:
            for doc_id, line in enumerate(f):
                doc = json.loads(line)
                text = doc.get('contents', '')
                tokens = self._tokenize(text)
                score = self._score(query_tokens, tokens)
                results.append((doc_id, score, doc))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


if __name__ == "__main__":
    # Test the BM25 retriever
    retriever = SimpleBM25Retriever('/root/autodl-fs/data/wiki-18.jsonl')
    results = retriever.search('What is Python?', k=5)
    for doc_id, score, doc in results:
        print(f"Doc {doc_id}, Score: {score:.4f}")
        print(f"  {doc.get('contents', '')[:100]}...")
        print()