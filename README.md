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

To use your own dataset:
* create a folder with the name of the dataset in folder "data".
* construct edges files in the folder, each with name "[NodeType1]\_[NodeType2]\_[RelationName (optional)].dat" and format of an edge in each line: "[Node1] [Node2]".
For example, filename "paper_conference" and content:
```
paper1 conf1
paper3 conf10
paper2 conf1
...
```