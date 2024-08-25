## Overview

- train.py: the core of our model, including the structure and the process of training.
- env.py, QLearning.py: the code about RL method
- GCN.py, layers.py: including the basic layers we used in the main model.
- dataset/: including the dataset:MUTAG, DD, NCI1, NCI109, PTC_MR, ENZYMES, PROTEINS.
  - 'RAW/': the original data of the dataset
  - adj.npy: the biggest Adjacency Matrix built from dataset
  - graph_label.npy: the label of every sub_graph
  - sub_adj.npy: the Adjacency Matrix of subgraph through sampling
  - features.npy: the pre-processed features of each subgraph


## Setting

1. setting python env using pip install -r requirements.txt
2. cd ./dataset &python transform.py --dataset MUTAG
3.  python train.py(all the parameters could be viewed in the train.py)

## Parameters
````
     --dataset DATASET
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
