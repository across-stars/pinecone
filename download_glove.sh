#!/bin/bash

out_dir=glove42
mkdir $out_dir && cd $out_dir
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip && unzip glove.42B.300d.zip && rm -r glove.42B.300d.zip
