# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json,re,os
import numpy as np
from sentence_transformers import SentenceTransformer
import datasets
import pytrec_eval
from sklearn.metrics.pairwise import cosine_similarity
def get_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores
class BGE_Retriever:
    def __init__(self,target='',cache='./cache'):
        doc_pairs = datasets.load_dataset('BRIGHT/', 'documents',cache_dir='cache')[target]
        self.doc_ids = []
        self.documents = []
        for dp in doc_pairs:
            self.doc_ids.append(dp['id'])
            self.documents.append(dp['content'])
        base_path='/home/qinxubo/data' if os.path.exists('/home/qinxubo/data') else "/root/autodl-tmp"
        self.model=SentenceTransformer(f'{base_path}/pretrained_models/bge-base-en-v1.5')
        batch_size = 512
        if not os.path.isdir(os.path.join(cache, 'doc_emb', target, f"long_{False}_128")):
            os.makedirs(os.path.join(cache, 'doc_emb', target, f"long_{False}_128"))
        cur_cache_file = os.path.join(cache, 'doc_emb', target, f"long_{False}_128", f'0.npy')
        if os.path.isfile(cur_cache_file):
            print("load cache file:",cur_cache_file)
            self.doc_emb = np.load(cur_cache_file,allow_pickle=True)
        else:
            print("building cache of ",target)
            self.doc_emb = self.model.encode(self.documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
            np.save(cur_cache_file, self.doc_emb)
        examples = datasets.load_dataset('BRIGHT/', 'examples',cache_dir='cache')[target]
        self.excluded_ids = {}
        for e in examples:
            self.excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            assert len(overlap)==0
    def retrieval_bge_with_reasoner(self,query_id,query):
        queries=[query]
        query_emb = self.model.encode(queries,show_progress_bar=False,batch_size=1, normalize_embeddings=True)
        scores = cosine_similarity(query_emb, self.doc_emb)
        scores = scores.tolist()
        return get_scores(query_ids=[query_id],doc_ids=self.doc_ids,scores=scores,excluded_ids=self.excluded_ids)
class Retriever_Full:
    def __init__(self):
        tasks=['biology','earth_science','economics','pony','psychology',
                                 'sustainable_living','aops','theoremqa_theorems',
                                 'theoremqa_questions']
        self.retriever_map={}
        for task in tasks:
            key=f"BRIGHT_bm25_reasoner_{task}"
            self.retriever_map[key]=BGE_Retriever(task)
    def retrieval_bge_with_reasoner(self,query_id,query,data_source):
        return self.retriever_map[data_source].retrieval_bge_with_reasoner(query_id,query)
base_path='/home/qinxubo/data' if os.path.exists('/home/qinxubo/data') else "/root/autodl-tmp"
model=SentenceTransformer(f'{base_path}/pretrained_models/bge-base-en-v1.5')
def compute_score(solution_str, ground_truth, data_source,format_reward=1):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    print(solution_str)
    query=solution_str
    if "<|im_start|>assistant" in query:
        query = query.split("<|im_start|>assistant", 1)[1]
    print("[Query Generated]:",query)
    label_item=json.loads(ground_truth)
    question=label_item['query']
    docs=label_item['pos_docs_list']
    question_emb=model.encode([question],normalize_embeddings=True)
    query_emb=model.encode([query],normalize_embeddings=True)
    doc_emb=model.encode(docs, normalize_embeddings=True)
    original_sim=question_emb @ doc_emb.T
    query_sim=query_emb@doc_emb.T
    advantage=np.sum(query_sim-original_sim).tolist()
    print("Got Advantage:",advantage)
    return advantage