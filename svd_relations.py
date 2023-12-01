from utils import INDICES
import numpy as np
from typing import List, Dict, Union, Tuple
import logging
import pinecone
import os
import json

relation_file_path = 'relations.txt'
output_folder = 'svd_outputs'
max_relations = 40

words = [
    'pain',
    'fire',
    'music',
    'professor',
    'rain',
    'cat',
    'inspiration',
    'revenge',
    'electricity',
    'algorithm',
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def SVD(n_values=10):
    relations = np.loadtxt(relation_file_path)
    U, S, Vt = np.linalg.svd(relations, full_matrices=False)
    reconstructed_matrix = U[:, :n_values] @ np.diag(S[:n_values]) @ Vt[:n_values, :]
    return reconstructed_matrix


def fetch_by_label(label: str, index) -> List:
    vector = index.fetch([label])['vectors']
    vector = vector[label]['values']
    return vector


def find_top_k_related(word: str, relations: np.ndarray, index, top_k=5, max_relations=40):
    vector = fetch_by_label(word, index)
    results = set()
    for i in range(len(relations)):
        relation = relations[i]
        vector_norm_l2 = np.linalg.norm(relation, ord=2)
        if vector_norm_l2 < 1e-2:
            continue
        target = list(vector + relation)
        retrieved_knns = index.query(vector=target, top_k=top_k)['matches']
        retrieved_knns = tuple([v['id'] for v in retrieved_knns])
        for knn in retrieved_knns:
            results.add(knn)
        if i >= max_relations:
            break
    return results


def main(SVD_rank):
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT")
            )
    if INDICES['glove'] not in pinecone.list_indexes():
       raise ValueError(f"Create GloVe index {INDICES['glove']} first")
    glove_index = pinecone.Index(INDICES['glove'])

    relations = SVD(SVD_rank)
    analogies = {}
    for word in words:
        logger.info(f'retrieving for {word}')
        retrieved_knns = find_top_k_related(word, relations, glove_index, top_k=3, max_relations=max_relations)
        analogies[word] = tuple(retrieved_knns)
        print(f'{word} done')

    print(analogies)
    output_file = os.path.join(output_folder, f"svd_rank_{SVD_rank}_max_{max_relations}.json")
    with open(output_file, 'w') as json_file:
        json.dump(analogies, json_file, indent=2)

main(SVD_rank=10)
