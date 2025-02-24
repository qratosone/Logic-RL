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
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
from datasets import concatenate_datasets
import argparse
from tqdm import tqdm
import json
import random
max_len=0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/delta_sim_eval')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    datasets_full_train=[]
    datasets_full_test=[]
    
    #data_source = load_dataset('BRIGHT/', 'examples',cache_dir=args.cache_dir)[args.task]
    for TARGET in ['biology','earth_science','economics','pony','psychology',
                                 'sustainable_living','aops','theoremqa_theorems',
                                 'theoremqa_questions']: #'stackoverflow','leetcode','robotics',
        dataset = datasets.load_dataset('BRIGHT/', 'examples',cache_dir='cache')[TARGET]
        doc_pairs = datasets.load_dataset('BRIGHT/', 'documents',cache_dir='cache')[TARGET]
        doc_ids = []
        documents = []
        doc_ids_dict={}
        for dp in doc_pairs:
            doc_ids.append(dp['id'])
            documents.append(dp['content'])
            doc_ids_dict[dp['id']]=dp['content']
        sampled_document=documents[0]
        #print(len(sampled_document))
        #if len(sampled_document)>400:
        #    print(TARGET)
        #dataset=dataset.filter(lambda example: len(example['query']) >= 8000)
        max_query_length=max([len(example['query']) for example in dataset])
        print(max_query_length)
        if max_query_length>8000:
            print(TARGET)
        train_dataset = dataset
        test_dataset = dataset
        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                query = example['query']
                
                qid=example["id"]
                excluded_ids=example['excluded_ids']
                gold_ids=example['gold_ids']
                docs=[doc_ids_dict[doc_id] for doc_id in gold_ids]
                #random.shuffle(docs)
                #doc_sample=docs[0]
                #print(doc_sample)
                #full_prompt=TEMPLATE.format(question=query,document=doc_sample)
                #assert len(full_prompt)<6000
                prompt = (f'Instructions:\n'
                  f'1. Identify the essential problem.\n'
                  f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
                  f'3. Draft an answer with as many thoughts as you have.\n'
                  f'Query:{query}\n\n')
                assert len(set(excluded_ids).intersection(gold_ids))==0
                solution = {
                    "query":query,
                    "pos_docs_list":docs
                }
                solution=json.dumps(solution)
                #print(solution['qrels'])
                #evaluator=pytrec_eval.RelevanceEvaluator(solution['qrels'],["ndcg_cut.10"])
                data = {
                    "prompt": [
                        {"role":"system","content":"You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "answer":solution
                }

                return data

            return process_fn


        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        #import pdb;pdb.set_trace()
        datasets_full_train.append(train_dataset)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        datasets_full_test.append(test_dataset)
        #import pdb;pdb.set_trace()

    local_dir = args.local_dir+"/unsloth"
    hdfs_dir = args.hdfs_dir
    
    datasets_full_train=concatenate_datasets(datasets_full_train)
    datasets_full_test=concatenate_datasets(datasets_full_test)
    #datasets_full_train=datasets_full_train.filter(lambda example: len(example['prompt'][0]['content']) >= 12000)
    #datasets_full_test=datasets_full_test.filter(lambda example: len(example['prompt'][0]['content']) >= 8000)
    print("final results:",len(datasets_full_train))
    datasets_full_train.to_parquet(os.path.join(local_dir, 'train.parquet'))
    datasets_full_test.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
