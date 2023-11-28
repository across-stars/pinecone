import os
import pinecone
import logging
import time

GLOVE_FILE = "glove42/glove.42B.300d.txt"
BUF_SIZE = 400_000 # ~154 lines
#BUF_SIZE = 260_000 # ~100 lines
#BUF_SIZE = 1040 # ~1 line
NUM_CHUNKS_TO_READ = None # set to None to read the whole file
clear_log = True
index_name = 'glove40'

if clear_log:
    with open('error_log.txt', 'w'):
        pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('error_log.txt', encoding='utf-8')
formatter = logging.Formatter('%(levelname)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

print(f"log file: error_log.txt")

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIRONMENT")
        )

if not os.path.exists(GLOVE_FILE):
    raise FileNotFoundError(f"check that {GLOVE_FILE} exists")

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=300, metric='cosine')
index = pinecone.Index(index_name)

def read_file_chunk(file, stop=None, buf_size=BUF_SIZE):
    cnt = 0
    while True:
        buffer = file.readlines(buf_size)
        if (not buffer) or (stop is not None and cnt >= stop):
            break
        yield buffer
        cnt += 1


embedding_file =  open(GLOVE_FILE, 'r')
missed_embedding_cnt = 0
line_cnt = 0
start_time_total = time.time()
for idx, embed_chunk in enumerate(read_file_chunk(embedding_file, NUM_CHUNKS_TO_READ)):
    chunk_len = len(embed_chunk)
    embeddings = []
    for embed in embed_chunk:
        embed = embed.rstrip("\n").split(" ")
        embeddings.append(
                (embed[0], list(map(float, embed[1:])))
        )
    try:
        # try upsert the whole chunk
        index.upsert(vectors=embeddings)
    except Exception as err:
        # trying to remove non=ascii characters from the chunk keys
        def is_ascii(string):
            return all(ord(c) < 128 for c in string)

        num_embed_in_chunk = len(embeddings)
        embeddings = [emb for emb in embeddings if is_ascii(emb[0])]
        try:
            index.upsert(vectors=embeddings)
            missed_embedding_cnt += num_embed_in_chunk - len(embeddings)
        except Exception as err:
            # try to upsert  by one
            for vec_idx, emb in enumerate(embeddings):
                try:
                    # try upsert by one
                    index.upsert(vectors=[emb])
                except Exception as err2:
                    logger.error(f"{str(err2)}\nchunk {idx} embed:\n{emb}")
                    missed_embedding_cnt += 1

    line_cnt += chunk_len
    if not idx % 100:
        logger.info(f"inserted {idx} chunk, {line_cnt} lines processed, {missed_embedding_cnt} embeds skipped so far")
    

embedding_file.close()
end_time_total = time.time()
logger.info(f"\ninsertion time: {end_time_total - start_time_total}\ntotal embeddgins uninserted due to err: {missed_embedding_cnt}")
logger.info(index.describe_index_stats())
