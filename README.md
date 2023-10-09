# Heterogeneous-text-graph-for-comprehensive-multilingual-sentiment-analysis
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


## Running training and evaluation