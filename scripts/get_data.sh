#!/bin/bash

mkdir -p data/corpora
mkdir -p data/word_embeddings

cd data/corpora
wget http://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip
unzip multinli_0.9.zip

wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip

cd ../word_embeddings
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
