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

import json
from pyserini import analysis
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
import datasets
import pytrec_eval
class BM25_Retriever:
    def __init__(self,target=''):
        self.analyzer=analysis.Analyzer(analysis.get_lucene_analyzer())
        doc_pairs = datasets.load_dataset('xlangai/bright', 'documents',cache_dir='cache')[target]
        self.doc_ids = []
        self.documents = []
        for dp in doc_pairs:
            self.doc_ids.append(dp['id'])
            self.documents.append(dp['content'])
        corpus = [self.analyzer.analyze(x) for x in self.documents]
        self.dictionary = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus))]
        self.bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(self.dictionary), normalize_queries=False, normalize_documents=False)
        examples = datasets.load_dataset('xlangai/bright', 'examples',cache_dir='cache')[target]
        self.excluded_ids = {}
        for e in examples:
            self.excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            assert len(overlap)==0
    def retrieval_bm25_with_reasoner(self,query_id,query):
        all_scores = {}
        query = self.analyzer.analyze(query)
        bm25_query = self.model[self.dictionary.doc2bow(query)]
        similarities = self.bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        for did, s in zip(self.doc_ids, similarities):
            all_scores[str(query_id)][did] = s
        for did in set(self.excluded_ids[str(query_id)]):
            if did!="N/A":
                all_scores[str(query_id)].pop(did)
        cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
        return all_scores
class Retriever_Full:
    def __init__(self):
        tasks=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions']
        self.retriever_map={}
        for task in tasks:
            key=f"BRIGHT_bm25_reasoner_{task}"
            self.retriever_map[key]=BM25_Retriever(task)
    def retrieval_bm25_with_reasoner(self,query_id,query,data_source):
        return self.retriever_map[data_source].retrieval_bm25_with_reasoner(query_id,query)
retriever_full=Retriever_Full()


def compute_score(solution_str, ground_truth, data_source):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if '<|endoftext|>' in solution_str:
        solution_str=solution_str.strip('<|endoftext|>')
    #print(solution_str)
    results_output=solution_str.split('<query>')
    punish_flag=False
    
    
    actual_length=1
    if len(results_output)>=2:
        reasoned_query="".join(results_output[1:])
        actual_length=len(results_output[1:])
        
    else:
        reasoned_query=results_output[0]
        print("No actual queries")
        print("------")
        punish_flag=True
    print(reasoned_query)
    
    qrels=ground_truth['qrels']
    qrels=json.loads(qrels)
    #print(qrels)
    qrels_filtered={}
    for qid in qrels:
        if qrels[qid] is not None:
            qrels_filtered[qid]=qrels[qid]
    for qid in qrels_filtered:
        for doc in qrels_filtered[qid]:
            qrels_filtered[qid][doc]=int(qrels_filtered[qid][doc])
    assert len(qrels_filtered)==1
    #print(qrels_filtered)
    evaluator=pytrec_eval.RelevanceEvaluator(qrels_filtered,["ndcg_cut.10"])
    #answer = extract_solution(solution_str=solution_str, method=method)
    full_metrics=0
    retriever=retriever_full[data_source]
    for qid in qrels_filtered:
        results=retriever.retrieval_bm25_with_reasoner(qid,reasoned_query)
        scores=evaluator.evaluate(results)
        full_metrics+=scores[qid]['ndcg_cut_10']
    if punish_flag:
        full_metrics/=10
    full_metrics/=actual_length
    print('ndcg@10:',full_metrics)
    return full_metrics/len(qrels)