# PGRA

The code for paper "Projected Graph Relation-Feature Attention Network for Heterogeneous Information Network Embedding."

## Requirements

We tested the code on:
* python 3.6
* pytorch 1.0.0
* networkx 2.3

other requirements:
* numpy
* tqdm

## Usage

First, extract "data.zip" file in folder "data". Then run the code using command:
```
python main.py -d [Dataset_Name] -m [Variant] -l [Regularized_value]
```
Available variants of the model are "DistMult", "TransH1" and "TransH2".
Available datasets are "DBLP", "Yelp", "Aminer", "DM" (Douban Movie).

In the default setting, the model will be trained on the first GPU.
