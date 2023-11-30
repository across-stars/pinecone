#!/bin/bash

out_dir=models
data_dir=data

mkdir -p $out_dir && cd $out_dir
# GloVe
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip && unzip glove.42B.300d.zip && rm -r glove.42B.300d.zip
# RelBERT
curl -L 'https://drive.google.com/uc?export=download&id=1z3UeWALwf6EkujI3oYUCwkrIhMuJFdRA&confirm=t' > gensim_model.bin.tar.gz && tar -xzf gensim_model.bin.tar.gz && mv gensim_model.bin relbert_model.bin && rm -r gensim_model.bin.tar.gz

mkdir -p ../$data_dir && cd ../$data_dir
wget https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset.zip && unzip analogy_test_dataset.zip && rm -r analogy_test_dataset.zip
wget https://raw.githubusercontent.com/taczin/SCAN_analogies/main/data/SCAN_dataset.csv