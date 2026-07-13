import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def load_corpus(corpus_path):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=1
    )
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path, use_fp16=False, device="cpu"):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16 and device == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean", device="cpu"):
    attention_mask = attention_mask.to(device)
    last_hidden_state = last_hidden_state.to(device)
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, device="cpu"):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list, is_query=True):
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(
            output.pooler_output,
            output.last_hidden_state,
            inputs['attention_mask'],
            self.pooling_method,
            self.device
        )
        if "dpr" not in self.model_name.lower():
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return query_emb


class DenseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = getattr(config, 'retrieval_method', 'e5')
        self.topk = getattr(config, 'retrieval_topk', 10)
        self.index_path = getattr(config, 'index_path', '')
        self.corpus_path = getattr(config, 'corpus_path', '')
        
        self.index = faiss.read_index(self.index_path)
        self.corpus = load_corpus(self.corpus_path)
        
        model_path = getattr(config, 'retrieval_model_path', '')
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=model_path,
            pooling_method=getattr(config, 'retrieval_pooling_method', 'mean'),
            max_length=getattr(config, 'retrieval_query_max_length', 256),
            use_fp16=getattr(config, 'retriever_use_fp16', False),
            device=getattr(config, 'device', 'cpu')
        )
        self.batch_size = getattr(config, 'retrieval_batch_size', 128)

    def search(self, query, num=None, return_score=False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def batch_search(self, query_list, num=None, return_score=False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results


class Config:
    def __init__(self, retrieval_method="bm25", retrieval_topk=10, 
                 index_path="./index/bm25", corpus_path="./data/corpus.jsonl",
                 dataset_path="./data", data_split="train", faiss_gpu=False,
                 retrieval_model_path="./model", retrieval_pooling_method="mean",
                 retrieval_query_max_length=256, retrieval_use_fp16=False,
                 retrieval_batch_size=128, device="cpu"):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retriever_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retriever_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.device = device


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if not request.topk:
        request.topk = config.retrieval_topk

    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


def get_retriever(config):
    if config.retrieval_method == "bm25":
        raise NotImplementedError("BM25 not supported in CPU version")
    else:
        return DenseRetriever(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch retriever (CPU support).")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="e5")
    parser.add_argument("--retriever_model", type=str, required=True)
    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    
    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean" if "e5" in args.retriever_name else "cls",
        retrieval_query_max_length=256,
        retrieval_use_fp16=args.faiss_gpu,
        retrieval_batch_size=512,
        device=args.device
    )

    retriever = get_retriever(config)
    uvicorn.run(app, host="0.0.0.0", port=8000)
