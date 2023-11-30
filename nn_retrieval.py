import os
import pinecone
import logging
import numpy as np
import pandas as pd
import json
import csv
from typing import List, Dict, Union
import argparse
from utils import DATASETS, INDICES


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Embedding():
    def __init__(self, data_path: str, prefix: str = 'all', top_k: int = 5):
        """
            data_path: path to the dataset
            prefix: a valid field in the 'prefix' column or 'all'
            index: name of the Pinecone index
            top_k: number of nearest neighbors to retrieve
        """
        self.data_path = data_path
        self.top_k = top_k
        logger.info(f'processinig {self.data_path}')
        # read the data
        if data_path.split('.')[-1] == 'jsonl':
            data_list = []
            with open(self.data_path, 'r') as file:
                for line in file:
                    json_data = json.loads(line)
                    data_list.append(json_data)

            self.df = pd.DataFrame(data_list)
        elif data_path.split('.')[-1] == 'csv':
            self.df = pd.read_csv(self.data_path)
        else:
            raise TypeError("pass .jsonl or .csv file")

        # TODO: more flexible way to handle prefixes
        if 'prefix' in self.df.columns:
            prefixes = set(self.df['prefix'].unique())
        else:
            prefixes = set(self.df['analogy_type'].unique())
        prefixes.add('all')
        if prefix not in prefixes:
            raise ValueError(f"No such prefix in the dataset!\nMust be one of {prefixes}")
        self.prefix = prefix


    def write_to_file(self, output_file: str):
        with open(output_file, 'w', newline='') as csv_file:
            fieldnames = ['base', 'target']
            fieldnames.extend([f'top{k}' for k in range(self.top_k)])
            fieldnames.extend(['k_order', 'k_score'])
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)
            writer.writerows(self.k_values)
        logging.info(f"Done! Results are written to {output_file}")

    def _fetch_top_matches(self):
        pass


    def compute_accuracy(self):
        accuracy_at_k = {}
        # compute accuracy @ k
        for k in range(self.top_k):
            correct = 0
            for row in self.k_values:
                if row[k+2] == row[1]:
                    correct += 1
            accuracy_at_k[k+1] = correct / len(self.k_values)
            if k > 0:
                accuracy_at_k[k+1] += accuracy_at_k[k]
        return accuracy_at_k




class Glove(Embedding):
    def __init__(self, data_path: str, prefix: str = 'all', index: str = INDICES['glove'], top_k: int = 5):
        super().__init__(data_path, prefix, top_k)
        print(self.prefix)
        partition = self._parse_data_by_prefix(self.df)
        self.k_values = self._fetch_top_matches(partition, index)

    def _parse_data_by_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """parses the data for quering the Glove"""
        if self.prefix != 'all':
            prefix = df[(df['prefix'] == self.prefix)]
        else:
            prefix = df
        prefix['analogy'] = prefix.apply(lambda row: row['choice'][row['answer']], axis=1)
        prefix = prefix.drop(['answer', 'choice', 'prefix'], axis=1)
        prefix['stem'] = prefix['stem'].map(lambda x: [x[0].lower(), x[1].lower()])
        prefix['analogy'] = prefix['analogy'].map(lambda x: [x[0].lower(), x[1].lower()])
        prefix[['w0', 'w1']] = prefix['stem'].apply(pd.Series)
        prefix[['w2', 'w3']] = prefix['analogy'].apply(pd.Series)
        prefix = prefix.drop(['stem', 'analogy'], axis=1)
        return prefix

    def _retrieve_corresponding_vectors(self, ws: List[str], index):
        """retrieves vectors by label"""
        vectors = index.fetch(ws)['vectors']
        vectors = {k: np.array(v['values']) for k, v in vectors.items()}
        vectors = (vectors[ws[0]], vectors[ws[1]], vectors[ws[2]], vectors[ws[3]])
        return vectors

    def _fetch_top_matches(self, partition: pd.DataFrame, index) -> List:
        """retrieve the top_k and compare with the target"""
        k_values = []
        for i in range(len(partition)):
            example = list(partition.iloc[i])
            try:
                vectors = self._retrieve_corresponding_vectors(example, index)
                vector_to_check = list(vectors[2] + vectors[1] - vectors[0])
                retrieved_vectors = index.query(vector=vector_to_check, top_k=self.top_k)['matches']
            except Exception as e:
                # TODO: handle this exception
                logger.error(f"Error {e} in retrieving vectors for {example}")
            # target value, [top_k values]
            row = [example[2], example[3]]
            row.extend([retrieved_vectors[k]['id'] for k in range(self.top_k)])
            k = 0
            while k < self.top_k:
                if retrieved_vectors[k]['id'] == example[3]:
                    #k_values.append((k, retrieved_vectors[k]['score']))
                    row.extend([k, retrieved_vectors[k]['score']])
                    break
                k += 1
            if k == self.top_k:
                # no that vector in the query result
                row.extend([None, None])
            k_values.append(row)
            if not i % 50:
                logger.info(f'processed {i} labels out of {len(partition)}')
        return k_values


class RelBERT(Embedding):
    skipped_lables = 0
    def __init__(self, data_path: str, prefix: str = 'all', index: str = INDICES['relbert'], top_k: int = 5):
        super().__init__(data_path, prefix, top_k)
        partition = self._parse_data_by_prefix(self.df)
        self.k_values = self._fetch_top_matches(partition, index)


    def _parse_data_by_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """parses the data for quering the Relbert"""
        if self.prefix != 'all':
            prefix = df[(df['prefix'] == self.prefix)]
        else:
            prefix = df
        prefix['analogy'] = prefix.apply(lambda row: row['choice'][row['answer']], axis=1)
        prefix = prefix.drop(['answer', 'choice', 'prefix'], axis=1)
        prefix['stem'] = prefix['stem'].map(lambda x: '__'.join([x[0].lower(), x[1].lower()]))
        prefix['analogy'] = prefix['analogy'].map(lambda x: '__'.join([x[0].lower(), x[1].lower()]))
        return prefix
    

    def _get_knn(self, labels: List[str], index) -> Union[List[Dict], List[None]]:
        """retrieves vectors by label"""
        vectors = index.fetch([labels['stem']])['vectors']
        if vectors == {}:
            # TODO: handle this exception
            logger.error(f"Error in retrieving vectors for {labels}")
            self.skipped_lables += 1
            return []
        vectors = [v['values'] for v in vectors.values()]
        retrieved_knns = index.query(vector=vectors, top_k=self.top_k)['matches']
        # print(retrieved_knns) for debugging -- to see what is retrieved
        return retrieved_knns


    def _fetch_top_matches(self, df: pd.DataFrame, index) -> List:
        """retrieve the top_k over the dataset"""
        k_values = []
        for i in range(len(df)):
            example = df.iloc[i]
            retrieved_vectors = self._get_knn(example, index)
            if retrieved_vectors == []:
                continue
            # stem, analogy, top0, top1, ...
            row = [example['stem'], example['analogy']]
            row.extend([v['id'] for v in retrieved_vectors])

            k = 0
            while k < self.top_k:
                # find which k is the match if any
                if retrieved_vectors[k]['id'] == example['analogy']:
                    row.extend([k, retrieved_vectors[k]['score']])
                    break
                k += 1
            if k == self.top_k:
                # no that vector in the query result
                row.extend([None, None])
            k_values.append(row)
            if not i % 50:
                print(f'processed {i} labels out of {len(df)}')
            # del it later
        print(f"total labels not found in the db {self.skipped_lables}")
        return k_values


def test_glove(args):
    # get index
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT")
            )
    if INDICES['glove'] not in pinecone.list_indexes():
       raise ValueError(f"Create GloVe index {INDICES['glove']} first")
    glove_index = pinecone.Index(INDICES['glove'])
    accuracy_at_k = {}
    for ds in args.dataset:
        if ds not in DATASETS:
            raise ValueError(f"Dataset {ds} not found")
        logging.info(f"Testing {ds} dataset")
        glove = Glove(DATASETS[ds], index=glove_index, top_k=args.top_k)
        create_dir(args.output_dir)
        glove.write_to_file(f"{args.output_dir}/glove_{ds}_top{args.top_k}.csv")
        accuracy_at_k[ds] = glove.compute_accuracy()
        logging.info(f"Accuracy @ k for {ds}: {accuracy_at_k[ds]}")

def test_relbert(args):
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT")
            )
    if INDICES['relbert'] not in pinecone.list_indexes():
       raise ValueError(f"Create RelBERT index {INDICES['relbert']} first")
    relbert_index = pinecone.Index(INDICES['relbert'])
    accuracy_at_k = {}
    for ds in args.dataset:
        if ds not in DATASETS:
            raise ValueError(f"Dataset {ds} not found")
        relbert = RelBERT(DATASETS[ds], index=relbert_index, top_k=args.top_k)
        create_dir(args.output_dir)
        relbert.write_to_file(f"{args.output_dir}/relbert_{ds}_top{args.top_k}.csv")
        accuracy_at_k[ds] = relbert.compute_accuracy()
        print(f"Accuracy @ k for {ds}: {accuracy_at_k[ds]}")

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
def main(args):
    if args.embedding == "glove":
        test_glove(args)
    elif args.embedding == "relbert":
        test_relbert(args)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--embedding", type=str, help="Embedding to test [glove, relbert]")
    argparse.add_argument("--dataset", nargs='+', type=str, default=[], help="Dataset to test [SAT, BATS, GOOGLE]")
    argparse.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    argparse.add_argument("--output_dir", type=str, default="outputs", help="Output directory name")
    args = argparse.parse_args()
    main(args)

## Usage: python nn_retrieval.py --embedding glove --dataset SAT BATS GOOGLE --output_dir results
