from utils import DATASETS, INDICES
from nn_retrieval import Glove, RelBERT, Embedding
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple
import logging
import pinecone
import os
import json

FIND_RELATIONS = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StructureAnalogy(Embedding):
    def __init__(self, data_path: str, glove_index: str, relbert_index: str, prefix: str = 'all', top_k: int = 7):
        super().__init__(data_path, prefix, top_k)
        self.partition = self._parse_data_by_prefix(self.df)
        self.glove_index = glove_index
        self.relbert_index = relbert_index
        self.skipped_lables = 0
    
    def _parse_data_by_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.prefix != 'all':
            prefix = df[(df['analogy_type'] == prefix)]
        else:
            prefix = df
        # debug
        prefix = df.head(100)

        # lower case all the words
        prefix['src_word'] = prefix['src_word'].str.lower()
        prefix['targ_word'] = prefix['targ_word'].str.lower()
        prefix['source'] = prefix['source'].str.lower()
        prefix['target'] = prefix['target'].str.lower()
        # replace spaces with underscores
        prefix['src_word'] = prefix['src_word'].str.replace(" ", "_")
        prefix['targ_word'] = prefix['targ_word'].str.replace(" ", "_")
        prefix['source'] = prefix['source'].str.replace(" ", "_")
        prefix['target'] = prefix['target'].str.replace(" ", "_")

        prefix.set_index('target', inplace=True)
        prefix['stem'] = prefix.index + "__" + prefix['targ_word']
        prefix['analogy'] = prefix['source'] + "__" + prefix['src_word']
        prefix.drop(['analogy_type'], axis=1, inplace=True)
        return prefix
    
    def _retrieve_diff_vector(self, ws: List[str], index):
        """retrieves vectors by label"""
        try:
            vectors = index.fetch(ws)['vectors']
            vectors = {k: np.array(v['values']) for k, v in vectors.items()}
            vector = vectors[ws[0]] - vectors[ws[1]]
            return vector
        except:
            logger.error(f"Error in retrieving vectors for {ws}")
            self.skipped_lables += 1
        return None
    
    def retrieve_relations(self, index: str):
        self.relations = []
        for i in range(len(self.partition)):
            example = [self.partition.iloc[i]['src_word'], self.partition.iloc[i]['targ_word']]
            retrieved_vectors = self._retrieve_diff_vector(example, index)
            if retrieved_vectors is not None:
                self.relations.append(retrieved_vectors)
    
    def _get_knn(self, labels: List[str], index: str) -> Union[List[Dict], List[None]]:
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
    
    def _fetch_top_matches(self, df: pd.DataFrame, index):
        retrieved_set = []
        for i in range(len(df)):
            attribute = {}
            example = df.iloc[i]
            retrieved_vectors = self._get_knn(example, index)
            attribute['stem'] = example['stem']
            attribute['analogy'] = example['analogy']
            attribute['retrieved'] = [{"id": retrieved_vectors[k]['id'], "score": retrieved_vectors[k]['score'], "word": retrieved_vectors[k]['id'].split("__")[0]} for k in range(self.top_k)]
            # if retrieved_set entity is same as stem entity, remove it
            attribute['retrieved'] = [r for r in attribute['retrieved'] if r['word'] != attribute['stem'].split("__")[0]]
            retrieved_set.append(attribute)
        return retrieved_set
    
    def _find_most_common_label(self, retrieved_set: List[Dict]) -> str:
        """find the most common label across all attributes in retrieved_set"""
        labels = []
        for attribute in retrieved_set:
            retrieved_labels = [label['word'] for label in attribute['retrieved']]
            labels.extend(retrieved_labels)
        most_common_label = max(set(labels), key=labels.count)
        return most_common_label
    
    def predict_label(self):
        self.outputs = {}
        for target in self.partition.index:
            retrieved_set = self._fetch_top_matches(self.partition.loc[target], self.relbert_index)
            # find the most common label across all attributes in retrieved_set
            label = self._find_most_common_label(retrieved_set)
            self.outputs[target] = {
                'true_label': self.partition.loc[target].iloc[0]['source'],
                'pred_label': label,
                'retrieved_set': retrieved_set
            }
        

def main():
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT")
            )
    if INDICES['relbert'] not in pinecone.list_indexes():
       raise ValueError(f"Create RelBERT index {INDICES['relbert']} first")
    if INDICES['glove'] not in pinecone.list_indexes():
       raise ValueError(f"Create GloVe index {INDICES['glove']} first")
    
    glove_index = pinecone.Index(INDICES['glove'])
    relbert_index = pinecone.Index(INDICES['relbert'])
    structure = StructureAnalogy(DATASETS['SCAN'], glove_index, relbert_index)
    structure.predict_label()
    # save the outputs to a file
    with open('outputs.json', 'w') as f:
        json.dump(structure.outputs, f, indent=2)

    if FIND_RELATIONS:
        structure.retrieve_relations(glove_index)
        # save the relations to a file (input list of relations to the model)
        with open('relations.txt', 'w') as f:
            for relation in structure.relations:
                f.write(" ".join(str(value) for value in relation) + "\n")


main()