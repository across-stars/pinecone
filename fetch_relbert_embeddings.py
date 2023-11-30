from relbert import RelBERT
import json
import os
from gensim.models import KeyedVectors
import pandas as pd
from utils import DATASETS

# Set this to True if you want to add the base vocab to the vocab
ADD_BASE_VOCAB = False

# Load the RelBERT model
model = RelBERT()

relbert_embeddings = []
relbert_words = []

for name,dataset in DATASETS.items():
    queries = []
    if name == "SCAN":
        df = pd.read_csv(dataset)
        # Create lists of target and source combinations
        target_list = [[target, targ_word] for target, targ_word in zip(df['target'], df['targ_word'])]
        source_list = [[source, src_word] for source, src_word in zip(df['source'], df['src_word'])]
        queries = source_list + target_list
    else:
        with open(dataset, 'r') as file:
            for line in file:
                data = json.loads(line)
                stem = data.get('stem', [])
                queries.append(stem)

    relbert_embeddings += model.get_embedding(queries, batch_size=32)
    
    # join the words in a query if there are multiple words
    words_list = []
    for query in queries:
        word1, word2 = query
        if " " in word1:
            word1 = word1.replace(" ", "_")
        if " " in word2:
            word2 = word2.replace(" ", "_")
        query = word1 + "__" + word2
        words_list.append(query)
    relbert_words += words_list

if ADD_BASE_VOCAB:
    base_model = KeyedVectors.load_word2vec_format('relbert_model.bin', binary=True)
    base_vocab = base_model.index_to_key
    print(f"Base vocab size: {len(base_vocab)}")

    relbert_vocab = set(relbert_words.keys())
    for word in base_vocab:
        if word not in relbert_vocab:
            relbert_words.append(word)
            relbert_embeddings.append(base_model[word])

print(f"Writing embeddings to file...")

MAX_NUM_WORDS = 100000
idx = 1
vocab = set()
with open("relbert_scan.txt", 'w', encoding='utf-8') as f:
    for word,embedding in zip(relbert_words, relbert_embeddings):
        if word in vocab:
            continue
        vocab.add(word)
        embedding_str = " ".join(str(value) for value in embedding)
        f.write(f"{word} {embedding_str}\n")
        if idx == MAX_NUM_WORDS:
            break

print(f"Vocab size: {len(relbert_words)}")
print("Done!")