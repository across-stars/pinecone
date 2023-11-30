import os
import pinecone
import logging
import numpy as np
import pandas as pd
import json
import csv
from typing import List

glove_parameters = {
    'data_path': '/scratch/gpfs/mb5157/cos597a/data/analogy_test_dataset/google/test.jsonl',
    'output_file': 'glove42/google_capitals.csv',
    'index_name': 'glove40',
}

relbert_parameters = {
    'data_path': 'relbert/scan_dataset.csv',
    'output_file': 'relbert/entities_knn.csv',
    'index_name': 'relbert',
}

#output_file = 'relbert/turney.csv'
#index_name = 'relbert'

class Glove():
    data_path = glove_parameters['data_path']
    prefixes = ('gram3-comparative', 'gram2-opposite', 'capital-world','gram4-superlative', 'gram7-past-tense', 'family',
            'gram6-nationality-adjective', 'city-in-state', 'currency', 'gram9-plural-verbs', 'gram1-adjective-to-adverb',
            'capital-common-countries', 'gram5-present-participle', 'gram8-plural')

    def __init__(self, prefix: str, index, top_k=5):
        """
            prefix could be:
            'gram3-comparative', 'gram2-opposite', 'capital-world','gram4-superlative', 'gram7-past-tense', 'family',
            'gram6-nationality-adjective', 'city-in-state', 'currency', gram9-plural-verbs', 'gram1-adjective-to-adverb',
            'capital-common-countries', 'gram5-present-participle', 'gram8-plural'
        """

        if prefix not in self.prefixes:
            raise ValueError("no such prefix in the dataset")
        self.prefix = prefix
        self.top_k = top_k
        # read the data
        data_list = []
        with open(self.data_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                data_list.append(json_data)

        df = pd.DataFrame(data_list)
        # prefixes = df['prefix'].unique()
        partition = self._parse_data_by_prefix(df, self.prefix)
        self.k_values = self._fetch_top_matches(partition, self.top_k, index)


    def _parse_data_by_prefix(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """parses the data for quering the Glove"""
        prefix = df[(df['prefix'] == prefix)]
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


    def _fetch_top_matches(self, partition: pd.DataFrame, top_k: int, index) -> List:
        """retrieve the top_k and compare with the target"""
        k_values = []
        for i in range(len(partition)):
            example = list(partition.iloc[i])
            vectors = self._retrieve_corresponding_vectors(example, index)
            vector_to_check = list(vectors[2] + vectors[1] - vectors[0])
            retrieved_vectors = index.query(vector=vector_to_check, top_k=top_k)['matches']
            # target value, top0 value, top1 value
            row = [example[3], retrieved_vectors[0]['id'], retrieved_vectors[1]['id']]
            k = 0
            while k < top_k:
                if retrieved_vectors[k]['id'] == example[3]:
                    #k_values.append((k, retrieved_vectors[k]['score']))
                    row.extend([k, retrieved_vectors[k]['score']])
                    break
                k += 1
            if k == top_k:
                # no that vector in the query result
                row.extend([None, None])
            k_values.append(row)

        return k_values


    def write_to_file(self, output_file):
        with open(output_file, 'w', newline='') as csv_file:
            fieldnames = ['target', 'top0', 'top1', 'k_order', 'k_score']
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)
            writer.writerows(self.k_values) 


class Relbert():
    data_path = relbert_parameters['data_path']

    def __init__(self, index, top_k=5):
        self.top_k = top_k

        df = pd.read_csv(self.data_path)
        self.k_values = self._fetch_top_matches(df, self.top_k, index)


    def _get_knn(self, labels: List[str], index):
        """retrieves vectors by label"""
        labels = ['__'.join(labels)]
        print(labels)
        vectors = index.fetch(labels)['vectors']
        print(vectors)
        raise ValueError("stop here")
        vectors = {k: np.array(v['values']) for k, v in vectors.items()}
        retrieved_vectors = index.query(vector=vector_to_check, top_k=top_k)['matches']
        return vectors


    def _fetch_top_matches(self, df: pd.DataFrame, top_k: int, index) -> List:
        """retrieve the top_k"""
        k_values = []
        for i in range(len(df)):
            example = df.iloc[i]
            example = [example['target'], example['targ_word']]
            vectors = self._get_knn(example, index)
            # target value, top0 value, top1 value
            row = [example[3], retrieved_vectors[0]['id'], retrieved_vectors[1]['id']]
            k = 0
            while k < top_k:
                if retrieved_vectors[k]['id'] == example[3]:
                    #k_values.append((k, retrieved_vectors[k]['score']))
                    row.extend([k, retrieved_vectors[k]['score']])
                    break
                k += 1
            if k == top_k:
                # no that vector in the query result
                row.extend([None, None])
            k_values.append(row)

        return k_values


    def write_to_file(self, output_file):
        with open(output_file, 'w', newline='') as csv_file:
            fieldnames = ['target', 'top0', 'top1', 'k_order', 'k_score']
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)
            writer.writerows(self.k_values) 


def main():
# get index
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT")
            )
    #if glove_parameters['index_name'] not in pinecone.list_indexes():
    #    raise ValueError(f"create index {glove_parameters['index_name']} first")
    #glove_index = pinecone.Index(glove_parameters['index_name'])

    if relbert_parameters['index_name'] not in pinecone.list_indexes():
        raise ValueError(f"create index {relbert_parameters['index_name']} first")
    relbert_index = pinecone.Index(relbert_parameters['index_name'])

    #glove_capitals = Glove('capital-world', glove_index)
    #glove_capitals.write_to_file(glove_parameters['output_file'])

    relbert_embeds = Relbert(relbert_index)


main()
