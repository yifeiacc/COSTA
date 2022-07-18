## Covariance-Preserving Feature Augmentation for Graph Contrastive Learning.

### Overview
This repo contains an example implementation for KDD'22 paper: **COSTA: Covariance-Preserving Feature Augmentation for Graph Contrastive Learning**. 
This code provides the multiview MV-COSTA. The SV-COSTA can be easily obtained by modifying the MV-COSTA.

### Overview

COSTA is a **feature augmentation** method that generates augmented samples in the feature space (latent space). It produced a bias-free and covariance-bounded augmentation to alleviate the bias problem in the typical graph augmentation (e.g., edge permutations). 

### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Reproduce our results
We provide several datasets to reproduce our results. We provide wandb logs to show the performance. For example

Cora: https://wandb.ai/yifeiacc/COSTA/runs/2gsyndrf/overview?workspace=user-yifeiacc

CiteSeer: https://wandb.ai/yifeiacc/COSTA/runs/31jrnccw?workspace=user-yifeiacc

The detailed settings (including hyper-parameters and GPUs) and the results can be found in these logs. You can directly checkout to the corresponding branch(commit).

### Usage
To run our code, just run the following
```
$ cd src 
$ python main.py --root path/to/COSTA/dir --dataset Cora --model COSTA --config COSTA_default.yaml
