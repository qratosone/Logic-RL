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
        self.model=SentenceTransformer('/root/autodl-tmp/pretrained_models/bge-base-en-v1.5')
        batch_size = 2048
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
retriever_full=Retriever_Full()
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<query>', 1),
        'answer_end': ('</query>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><query>...</query>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed
def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<query>(.*?)</query>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid query tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str
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
    query, solution_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{solution_str}")
    format_correct = validate_response_structure(solution_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")
    #if not format_correct:
    #    return format_score
    if query is None:
        query=solution_str
    print("[Query Generated]:",query)
    qrels=json.loads(ground_truth)['qrels']
    qrels_filtered={}
    for qid in qrels:
        if qrels[qid] is not None:
            qrels_filtered[qid]=qrels[qid]
    for qid in qrels_filtered:
        for doc in qrels_filtered[qid]:
            qrels_filtered[qid][doc]=int(qrels_filtered[qid][doc])
    assert len(qrels_filtered)==1
    evaluator=pytrec_eval.RelevanceEvaluator(qrels_filtered,["ndcg_cut.10"])
    #answer = extract_solution(solution_str=solution_str, method=method)
    full_metrics=0
    retriever=retriever_full.retriever_map[data_source]
    for qid in qrels_filtered:
        results=retriever.retrieval_bge_with_reasoner(qid,query)
        scores=evaluator.evaluate(results)
        full_metrics+=scores[qid]['ndcg_cut_10']
    #full_metrics/=len(query)
    print('ndcg@10:',full_metrics)
    print("Actual metrics:",full_metrics)
    return full_metrics/len(qrels)+format_score