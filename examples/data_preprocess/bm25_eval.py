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

from verl.utils.hdfs_io import copy, makedirs
import argparse
from tqdm import tqdm
import pytrec_eval



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/bm25_eval')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    num_few_shot = 5
    #data_source = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]
    TARGET='biology'
    dataset = datasets.load_dataset('xlangai/bright', 'examples',cache_dir='cache')[TARGET]
    doc_pairs = datasets.load_dataset('xlangai/bright', 'documents',cache_dir='cache')[TARGET]
    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])

    train_dataset = dataset
    test_dataset = dataset

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            query = example['query']
            qid=example["id"]
            excluded_ids=example['excluded_ids']
            gold_ids=example['gold_ids']
            assert len(set(excluded_ids).intersection(gold_ids))==0
            question = query + ' ' + instruction_following
            solution = {
                "qrels":{qid:{}}
            }
            for gold_id in gold_ids:
                solution['qrels'][qid][gold_id]=1
            #print(solution['qrels'])
            evaluator=pytrec_eval.RelevanceEvaluator(solution['qrels'],["ndcg_cut.10"])
            data = {
                "data_source": f"BRIGHT_bm25_reasoner_{TARGET}",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split':split,
                    "idx":idx,
                    'query': query,
                    'qid': qid
                }
            }

            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir+"/"+TARGET
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
