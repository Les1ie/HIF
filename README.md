![image](https://github.com/Les1ie/HIF/assets/77771959/462bf947-f456-4bfb-9abc-7ab7f54adf22)# HIF
Heterogeneous GNN Driven Information Prediction Framework

## Description
Our HIF is implemented mainly based on the following libraries (see the README file in source code folder for more details):

- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://www.pytorchlightning.ai/
- DGL: https://www.dgl.ai/
- PytorchNLP: https://github.com/PetrochukM/PyTorch-NLP
- Networkx: https://networkx.org/

### File Tree

Our main project file structure and description:

```python
HIF
├─ README.md
├─ environment.yaml       # our environment list
├─ data
│    ├─ processed       # processed data will be made by program
│    ├─ raw       # our raw data
├─ models       # package of models (HIF and its variants)
│    ├─ __init__.py
│    ├─ classfier.py
│    ├─ hetero.py
│    ├─ predictor.py
│    ├─ test.py
├─ nn       # package of neural network layers
│    ├─ coattention
│    ├─ conv
│    ├─ functional
│    ├─ metrics
│    ├─ nlp
│    ├─ tcn
│    ├─ time
│    ├─ __init__.py
│    ├─ conv.py	# graph convolution layer
│    ├─ embedding.py       # embedding module
│    ├─ noise.py
│    ├─ positional_encoding.py
│    ├─ reinforcements.py
│    └─ readout.py	# output layer
├─ run.py	# running entrance
└─ utils	# utils of drawing, dataloading, tensor and parameter propcessing
       ├─ data
       ├─ draw
       ├─ graph_op
       ├─ __init__.py
       ├─ additive_dict.py
       ├─ ckpt.py
       ├─ clean_lightning_logs.py
       ├─ group_tensorborad_data.py
       ├─ indexer.py
```

### Installation

Installation requirements are described in `environment.txt`

- Create Conda Environment:

  Open a terminal or command prompt and run the following command:

  ```bash
  conda env create -f environment.yml
  ```

- Activate Conda Environment:

  After the installation is complete, activate the newly created Conda environment using the following command:

  ```bash
  conda activate <environment_name>
  ```
  
- Verify Environment Installation:

  You can verify whether the environment is correctly installed:

  ```bash
  conda list
  ```

## Usage

Before it starts running, please make sure that HIF is located in the `/root/hif` directory

Then, get helps for all parameters of data processing, training and optimization:

```bash
python run.py  --help
```

Run:

```bash
python run.py --paramater value
```

### Experiment Settings

Here we list some basic settings of parameters for your reference:

**Basic settings**

|      Parameter        |       Value       | Description                                                                                  |
| :------------------:  | :---------------: | -------------------------------------------------------------------------------------------- |
|       gpus            |        1          | Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node.   |
|   time_loss_weig      |       5e-5        | Wights of time nodes similarity.                                                             |
|     num_workers       |        4          | Number of workers.                                                                           |
|      num_heads        |        16         | Number of attention heads.                                                                   |
|   log_every_n_steps   |        5          | How often to log within steps.                                                               |
|     batch_size        |        64         | Batch Size.                                                                                  |
|        name           |       SSC         | Datasets name.                                                                               |
|     observation       |        1.0        | Observation time.                                                                            |
|        sample         |       0.05        | Sample.                                                                                      |
|accumulate_grad_batches|        1          | Accumulates grads every k batches or as set up in the dict.                                  |
|      in_feats         |       256         | Dimension of inputs.                                                                         |
|     learning_rate     |      5e-3         | Learning rate to optimize parameters.                                                        |
|     weight_decay      |       5e-3        | Weight decay.                                                                                |
|       l1_weight       |        0          | L1 loss.                                                                                     |
|       patience        |       35          | Patience of early stopping.                                                                  |
|    num_time_nodes     |        6          | Number of time nodes.                                                                        |
|     soft_partition    |        2          | Time node soft partition size.                                                               |
|      num_gcn_layers   |        2          | The number of gcn layers.                                                                    |
|       readout         |        ml         | The readout module.                                                                          |
|  num_readout_layers   |        3          | The number of internal layers of readout module.                                             |
|  time_decay_pos       |      all          | Position of time decay.                                                                      |
|  time_module          |      transformer  | Time embedding module.                                                                       |
|num_time_module_layers |      4            | The number of layers of time module, such as rnn layers or transformer encoder layers.       |
|       dropout         |     0.3           | Dropout.                                                                                     |
|       dropout_edge    |       0           | Dropout edges.                                                                               |
|    noise_weight       |       0           | Weight of noise.                                                                             |
|    noise_rate         |       1           | Coverage rate of noise.                                                                      |
|    noise_dim          |       1           | Dimension of noise.                                                                          |
|        hop            |       1           | Hops of sampled follower.                                                                    |
|        alpha          |       10          | Weight to adjust sampled follower                                                            |
|        beta           |       100         | Minimum number of sampled follower.                                                          |
|       source_base     |       4           | Weight of source nodes.                                                                      |
|     reposted_base     |       5           | Weight of reposted nodes.                                                                    |
|     leaf_base         |       6           | Weight of leaf nodes.                                                                        |
|     method            |       bfs         | Method to sample follower.                                                                   |
|   gradient_clip_val   |       1           | The value at which to clip gradients.                                                        |
|   max_epochs          |       300         | Stop training once this number of epochs is reached.                                         |
|enable_progress_bar    |      False        | Whether to enable to progress bar by default.                                                |
|enable_model_summar    |      False        | Whether to enable model summarization by default.                                            |
