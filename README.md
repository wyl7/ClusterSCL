# ClusterSCL
The pytorch implementation of ClusterSCL: Cluster-Aware Supervised Contrastive Learning on Graphs ([WWW 2022](https://www2022.thewebconf.org/)).

ClusterSCL is a contrastive learning scheme for supervised learning of node classification models on the datasets with large intra-class variances and high inter-class similarities. The main though is not restricted to the node classification task.

You can read our [paper](https://xiaojingzi.github.io/publications/WWW22-Wang-et-al-ClusterSCL.pdf) for details on this algorithm.

Requirements
====
Create a virtual environment first via:
```
$ conda create -n your_env_name python=3.8.5
```

Install all the required tools using the following command:
```
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

$ conda install -c dglteam dgl-cuda11.1

$ pip install -r requirements.txt
```

Overview
====
This repo covers an reference implementation for the following methods in PyTorch, using Pubmed as an illustrative example:

(1) ClusterSCL.

(2) Supervised Contrastive Learning (SupCon). [paper](https://arxiv.org/abs/2004.11362)

You can run train_ce.py for an end-to-end supervised learning of a GNN-based classification model using cross-entropy loss. 
You can run train_supcon.py or train_clusterscl.py for two-stage learning of the GNN-based classification model.

Specifically, the repository is organized as follows:
* `data/` contains the data files from https://github.com/tkipf/gcn.

* `networks/` contains the implementation of a GAT backbone.

* `requirements.txt` is used to install the required tools.
 
* `util.py` is used for loading and pre-processing the dataset.
 
* `losses.py` is the implementation of SupCon loss from https://github.com/HobbitLong/SupContrast.
 
* `elboCL.py` is the implementation of ClusterSCL loss.

Loss function
====
The loss function ClusterSCL in elboCL.py takes a batch of node embeddings (L2 normalized) and the nodes' labels as input, and return the loss value.

Running the code
====
(1) Create a folder to store results.
```
$ mkdir save
```

(2) To run the example, execute:
```
$ sh run_pubmed.sh
```

Notes: the optimal hyper-parameters could be somewhat different under different environments (e.g., different devices and different versions of PyTorch), but the experimental conclusion will not be affected. You can use the suggested method introduced in our paper to choose the combination of hyper-parameters.

Reference
====
If you make use of our idea in your work, please cite the following paper:
```
 @inproceedings{Wang2021decoupling,
     author = {Yanling Wang and Jing Zhang and Haoyang Li and Yuxiao Dong and Hongzhi Yin and Cuiping Li and Hong Chen},
     title = {Cluster-Aware Supervised Contrastive Learning on Graphs},
     booktitle = {WWW},
     year = {2022}
   }
```
