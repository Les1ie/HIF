# HIF
Heterogeneous GNN Driven Information Prediction Framework

## Description
Our HIF is implemented mainly based on the following libraries (see the README file in source code folder for more details):

- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://www.pytorchlightning.ai/
- DGL: https://www.dgl.ai/
- PytorchNLP: https://github.com/PetrochukM/PyTorch-NLP
- Networkx: https://networkx.org/

### File Tree

Project file structure and description:

```python
HERI-GCN
├─ README.md
├─ environment.yaml # our environment list
├─ dataloading	# package of dataloading
│    ├─ __init__.py
│    ├─ datamodule.py	
│    └─ dataset	# package of dataset class
│           ├─ WeiboTopicDataset.py	# process data of csv format 
│           └─ __init__.py
├─ hooks.py	# hooks of model
├─ model	# package of models (HERI-GCN and its variants)
│    ├─ PopularityPredictor.py
│    ├─ __init__.py
├─ nn	# package of neural network layers
│    ├─ __init__.py
│    ├─ conv.py	# graph convolution layer
│    └─ readout.py	# output layer
├─ requirements.txt	
├─ run.py	# running entrance
└─ utils	# utils of drawing, dataloading, tensor and parameter propcessing
       ├─ __init__.py
       ├─ arg_parse.py
       ├─ dataloading.py
       ├─ drawing.py # heterogenous graph drawing
       ├─ output.py
       └─ utils.py
```
