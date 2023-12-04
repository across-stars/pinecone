This is the project made for [**COS597A Long Term Memory in AI - Vector Search and Databases** class in Princeton](https://edoliberty.github.io//vector-search-class-notes/) by Akshara Prabhakar (aksh555) and Margarita Belova (across-stars)

The **VectorDB project.pdf** presentation explains the underlying principle for searching analogies.<br>
The svd_outputs, outputs folders contain examples of analogies retrieved by scripts.

# Word & System Analogies Retrieval

## Setup
1. Run ```download.sh``` to download the analogy datasets, and GloVe and RelBERT embedding models. As a result, you'll get text file with embeddgins.
2. Run ```created_pincone_db.py``` to upsert the dataset into Pinecone.
parameters to set in ```created_pincone_db.py```: 
 * INPUT_FILE -- path to the text file with embeddings
 * dim -- 1024 for Relbert embeddings, 300 for Glove.
 * BUF_SIZE -- chunk size in bytes to read the text file.
 * index_name -- your index name in Pinecone

Make sure you have yourPINECONE_API_KEY and PINECONE_ENVIRONMENT set in the system environment to connect to Pinecone.

## Word Analogy
1. Change the path in ```utils.py``` to point to the data directory containing the datasets (BATS, SAT, GOOGLE, SCAN) -- the directory should be created on the previous step by the ```download.sh``` script.

2. Run ```fetch_relbert_embeddings.py``` to  get the embeddings of the task instances in all the datasets.

```nn_retrieval.py``` implements nearest neighbor retrieval. 

```bash
python nn_retrieval.py --embedding <glove> --dataset SAT BATS GOOGLE --output_dir <results>
```
Specify the list of datasets to evaluate, and the embedding to use. The results are stored in the output directory and the metric ```accuracy@k``` is computed. Or run ```python nn_retrieval.py --help``` to see the full list of parameters.

As a result, you'll get the file in the format:<br>
base,target,top0,top1,top2,...,k_order,k_score<br>

where base -- source word, target -- analogy to it, k_order -- which order the target took out of the top_k retrieved nearest neighbors, k_score -- the cosine similarity between base and target.
(check the **output** folder in the repo too see the example output for different datasets and embeddings)

## System Analogy
To evaluate system analogies, run ```structure_analogy.py``` to evaluate on SCAN.

## SVD
We have a library of 443 relations (relations.txt) acting as translations in the GLOVE space. The library is the result of the structure_analogy step (with some modifications).<br>
To get novel analgoies, launch:
```python sdv_relations.py```
The script runs the SVD over the library realtions in order to remove the noise, excludes relation vectors with L2 norm < 1e-2, and runs knn over no more then ```max_relations``` relations (default is 40).<br>
Parameters:
* words --  list inside the script with the words you want to find analogies for. The output is the file with lists of the most analogical systems.
* SVD_rank -- rank of the SVD decompositon.

**Check the output example in the svd_outputs folder.**
