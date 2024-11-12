# README

## Introduction
Our model is divided into three modules: heterogeneous graph network and subgraph contrastive learning. You can find specific descriptions in the corresponding directories.

## Requirement
Utilize pip install -r requirements.txt to set 


## Subgraph Contrastive Learning

### Overview

- train.py: the core of this module, including the structure and the process of training.
- env.py, QLearning.py: the code about the Contrastive Learning method
- GCN.py, layers.py: including the basic layers.
- dataset/: YourData
  - 'RAW/': the original data of the dataset
  - adj.npy: the biggest Adjacency Matrix built from dataset
  - graph_label.npy: the label of every sub_graph
  - sub_adj.npy: the Adjacency Matrix of subgraph through sampling
  - features.npy: the pre-processed features of each subgraph


## Setting

1. cd ./dataset &python transform.py --dataset YourData
2. python train.py (all the parameters could be viewed in the train.py)

## Parameters
````
     --dataset YourData
     --num_info NUM_INFO
     --lr LR (learning_rate)
     --max_pool MAX_POOL
     --momentum MOMENTUM
     --num_epoch NUM_EPOCH
     --batch_size BATCH_SIZE
     --sg_encoder SG_ENCODER(GIN, GCN, GAT, SAGE)
     --MI_loss MI_LOSS
     --start_k START_K
````


## Citation
If you take advantage of the HSNPL model in your research, please cite the following in your manuscript:
```
@article{chen2024heterogeneous,
  title={Heterogeneous Subgraph Network with Prompt Learning for Interpretable Depression Detection on Social Media},
  author={Chen, Chen and Li, Mingwei and Li, Fenghuan and Chen, Haopeng and Lin, Yuankun},
  journal={arXiv preprint arXiv:2407.09019},
  year={2024}
}
```
