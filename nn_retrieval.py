import os
import pinecone
import logging
import numpy as np
import pandas as pd
import json
import csv
from typing import List, Tuple

data_path = '/scratch/gpfs/mb5157/cos597a/data/analogy_test_dataset/google/test.jsonl'
output_file = 'glove42/google_capitals.csv'
index_name = 'glove40'

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
#file_handler = logging.FileHandler('log.txt', encoding='utf-8')
#formatter = logging.Formatter('%(levelname)s %(message)s')
#file_handler.setFormatter(formatter)
#logger.addHandler(file_handler)

# read the data
data_list = []
with open(data_path, 'r') as file:
    for line in file:
        json_data = json.loads(line)
        data_list.append(json_data)

df = pd.DataFrame(data_list)


def get_data_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """parses the data"""
    prefix = df[(df['prefix'] == prefix)]
    prefix['analogy'] = prefix.apply(lambda row: row['choice'][row['answer']], axis=1)
    prefix = prefix.drop(['answer', 'choice', 'prefix'], axis=1)
    prefix['stem'] = prefix['stem'].map(lambda x: [x[0].lower(), x[1].lower()])
    prefix['analogy'] = prefix['analogy'].map(lambda x: [x[0].lower(), x[1].lower()])
    prefix[['w0', 'w1']] = prefix['stem'].apply(pd.Series)
    prefix[['w2', 'w3']] = prefix['analogy'].apply(pd.Series)
    prefix = prefix.drop(['stem', 'analogy'], axis=1)
    return prefix


prefixes = df['prefix'].unique()
# process the capital analogies
capitals = get_data_prefix(df, prefixes[2]) # 'capital-world'

# get index
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIRONMENT")
        )
if index_name not in pinecone.list_indexes():
    raise ValueError("create index first")
index = pinecone.Index(index_name)

#def query_top_k(word_id, top_k: int = 10):
#    query = index.query(
#        id=word_id,
#        top_k=top_k,
#    )
#    return query


def retrieve_corresponding_vectors(ws: List[str]):
    """retrieves vectors by label"""
    vectors = index.fetch(ws)['vectors']
    vectors = {k: np.array(v['values']) for k, v in vectors.items()}
    vectors = (vectors[ws[0]], vectors[ws[1]], vectors[ws[2]], vectors[ws[3]])
    return vectors


k_values = []
top_k = 5
# retrieve the top_k and compare with the target
for i in range(len(capitals)):
    example = list(capitals.iloc[i])
    vectors = retrieve_corresponding_vectors(example)
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

with open(output_file, 'w', newline='') as csv_file:
    fieldnames = ['target', 'top0', 'top1', 'k_order', 'k_score']
    writer = csv.writer(csv_file)
    writer.writerow(fieldnames)
    writer.writerows(k_values)
