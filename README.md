# Heterogeneous-text-graph-for-comprehensive-multilingual-sentiment-analysis
[![DOI](https://zenodo.org/badge/702031299.svg)](https://zenodo.org/doi/10.5281/zenodo.10450316)

In this work, we propose a multilingual sentiment analysis approach based on graph convolution network. We create a single heterogeneous text graph to model the entire data of a multilingual corpus.
Then, we adopt a slightly deep graph convolution network to model the entire graph and to learn words and documents representations based on heterogeneous information.
Our work is motivated by the success of the proposed TextGCN [Yao et al. (2019)](https://arxiv.org/abs/1809.05679).

A portion of our code has been sourced from the repositories [TextGCN](https://github.com/yao8839836/text_gcn) and [GNN-for-text-classification](https://github.com/zshicode/GNN-for-text-classification).

## Requirements

* Python 3.7
* pandas 1.2.4
* numpy 1.21.5
* datasets from Hugging face
* re
* pickle 4.0
* scipy 1.5.2
* torch 1.11.0
* dgl 0.8.2
* Aligned word embedding [MUSE](https://github.com/facebookresearch/MUSE)

## Running training and evaluation

First, we provide the row data (Amazon reviews dataset, IMDB, Allociné, and Muchocine). Then, we start building the multilingual sentiment analysis corpura:
1. `cd ./data_transform`
2. Run `python amazon_transform.py` for multilingual amazon reviews dataset (NB: Select the set of languages) or `python MR_transform.py` for multilingual movie reviews dataset.
Start processing the datasets
3. `cd ./preprocess`
4. Run `python remove_words.py <dataset>`
5. Add the aligned word embedding and word similarity files to their corresponding directories (just the files of the languages used)
3. Then, run `python build_graph.py <dataset>`
4. `cd ../`
5. Finally, run `main.py` to train and evaluate the proposed model.


