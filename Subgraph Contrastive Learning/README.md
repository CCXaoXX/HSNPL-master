## Overview

- train.py: the core of our model, including the structure and the process of training.
- env.py, QLearning.py: the code about Contrastive Learning method
- GCN.py, layers.py: including the basic layers we used in the main model.
- dataset/: 
  - 'RAW/': the original data of the dataset
  - adj.npy: the biggest Adjacency Matrix built from dataset
  - graph_label.npy: the label of every sub_graph
  - sub_adj.npy: the Adjacency Matrix of subgraph through sampling
  - features.npy: the pre-processed features of each subgraph


## Setting

1. cd ./dataset &python transform.py --dataset YourData
2.  python train.py(all the parameters could be viewed in the train.py)

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
