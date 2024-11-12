# README

## Requirement
Utilize 'pip install -r requirements.txt' to set 

## Heterogeneous Graph Neural Network

### Easy Run

```
cd ./HeterogeneousGNN/model/code/
python train.py
```

### Prepare your dataset

```
cd ./HeterogeneousGNN/
```

You may change the dataset by modifying the variable "dataset = 'example'" at the top of the code "train.py" or use arguments (see train.py). 

The following files are required:

    ./model/data/YourData/
        ---- YourData.cites                // the adjacencies
        ---- YourData.content.text         // the features of texts
        ---- YourData.content.entity       // the features of entities
        ---- YourData.content.topic        // the features of topics
        ---- train.map                     // the index of the training node
        ---- vali.map                      // the index of the validation nodes
        ---- test.map                      // the index of the testing nodes

The format is as follows:

- **YourData.cites**

  Each line contains an edge:     "idx1\tidx2\n".        eg: "98	13"

- **YourData.content.text**

  Each line contains a node:    "idx\t[features]\t[category]\n", note that the [features] is a list of floats with '\t' as the delimiter.      eg:    "59	1.0	0.5	0.751	0.0	0.659	0.0	computers"
  If used for multi-label classification,  [category] must be one-hot with space as a delimiter,       eg:   "59	1.0	0.5	0.751	0.0	0.659	0.0	0 1 1 0 1 0"

 - **YourData.content.entity**

   Similar with .text, just change the [category] to "entity".		eg: "13	0.0	0.0	1.0	0.0	0.0	entity"

 - **YourData.content.topic**

   Similar with .text, just change the [category] to "topic".		eg: "64	0.10	1.21	8.09	0.10	topic"

 - ***.map**

   Each line contains an index:     "idx\n".              eg:  "98"

You can see the example in ./model/data/example/*

----

A simple data preprocessing code is provided. Successfully running it requires a token of [tagme](https://sobigdata.d4science.org/web/tagme/tagme-help "TagMe")'s account  (my token is provided in tagme.py, but may be invalid in the future), [Wikipedia](https://dumps.wikimedia.org/ "WikiPedia")'s entity descriptions, and a word2vec model containing entity embeddings. 

Then, you should prepare a data file like ./data/example/example.txt, whose format is:         "[idx]\t[category]\t[content]\n". 

Finally, modify the variable "dataset = 'example'" at the top of the following codes and run:

```
python tagMe.py
python build_network.py
python build_features.py
python build_data.py
```

### As GNN

If you just wanna use the model as a graph neural network, you can just prepare some files following the above format:

     ./model/data/YourData/
        ---- YourData.cites                // the adjacencies
        ---- YourData.content.*            // the features of *, namely node_type1, node_type2, ...
        ---- train.map                     // the index of the training node
        ---- vali.map                      // the index of the validation nodes
        ---- test.map                      // the index of the testing nodes

And change the   "load_data()"  in ./model/code/utils.py

```
type_list = [node_type1, node_type2, ...]
type_have_label = node_type
```

## Subgraph Contrastive Learning

### Run

```
cd ./SubgraphCL/
cd ./dataset/ 
python transform.py --dataset YourData
python train.py --all the parameters can be viewed in the train.py
```
### Introduction
```
- train.py: the core of this module, including the structure and the process of training
- env.py, QLearning.py: the code about the Contrastive Learning part
- GCN.py, layers.py: including the basic layers
- dataset/: YourData, available through the previous section
  - 'RAW/': the original data of the dataset
  - adj.npy: the biggest Adjacency Matrix built from the dataset
  - graph_label.npy: the label of every sub_graph
  - sub_adj.npy: the Adjacency Matrix of subgraph through sampling
  - features.npy: the pre-processed features of each subgraph
```

### Parameters
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

See the codes for more details.

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
