# Dataset

We've provided a pre-processed dataset, which can be used for easy runs.

## HeterogeneousGNN

+ `depression.txt`: Filtered depression dataset
+ `depression.conten`t: All kinds of features, such as prompt **scale**, **time** distribution, **emo**ticon ratio, **sent**iment word ratio, first-**person** singular and plural ratio, original and retweeted tweets(**twt**), following and follower lists(**net**)
+ Those files belong to `HeterogeneousGNN/model/data/YourData/`

## SubgraphCL

+ After the previous steps, some files are required for the next module, including: 
  + `.content` and `.emb2`(which can be found after the HeterogeneousGNN module)
  + `depression.txt`
  + `mapindex.txt`
  + `.cites`
+ Then utilize `convert.py` for converting data, and you can also decide what features you need in this file.
+ Converted files are supposed to be placed on `SubgraphCL/dataset/YourData/RAW/`
+ Finally, you might need to run `SubgraphCL/dataset/transform.py`

## Tools

Due to the huge amount of pre-processing work, we've provided pre-processed data. However, some useful tools might be needed if you are willing to regenerate features.

