# Word & System Analogies Retrieval

## Setup
Run ```download.sh``` to download the analogy datasets, and GloVe and RelBERT embedding models

## Word Analogy
Change the path in ```utils.py``` to point to the data directory containing the datasets.

Run ```fetch_relbert_embeddings.py``` to  get the embeddings of the task instances in all the datasets.

TODO: Pinecone db

```nn_retrieval.py``` implements nearest neighbor retrieval. 

```bash
python nn_retrieval.py --embedding <glove> --dataset SAT BATS GOOGLE --output_dir <results>
```
Specify the list of datasets to evaluate, and the embedding to use. The results are stored in the output directory and the metric ```accuracy@k``` is computed.

## System Analogy
To evaluate system analogies, run ```structure_analogy.py``` to evaluate on SCAN.