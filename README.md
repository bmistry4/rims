# Recurrent Independent Mechanisms (RIMs)
Reimplementation of [Recurrent Independent Mechanisms (Goyal et al. 2021)](https://openreview.net/pdf?id=mLcmdlEUxy-) 
using Pytorch. 
This project was done purely for learning purposes. 

RIMs are a RNN based architecture that learns to modularise the input dynamics into independent mechanisms to improve
generalisation and modelling of long-term dependencies. 
These mechanisms are reusable modules (where each module is represented by a subset of a LSTM’s weight matrices). 
The modules are independent from each other, only interacting sparsely via attention. 
Specifically, 
 
1. The RIMs will compete for access to the input, from which only a subset (the top-k) will be selected.
2. The selected RIMs update their knowledge with respect to the input. 
3. The RIMs communicate via information sharing. Only the top-k RIMs are allowed to access information from the other RIMs. 

# Implementation notes
Our implementation take influences from the [official RIMs repo](https://github.com/anirudh9119/RIMs) and an implementation
by [dido1998](https://github.com/dido1998/Recurrent-Independent-Mechanisms). 

The experiments are logged on [W&B](https://docs.wandb.ai/) and the code is implemented using the 
[Pytorch Lightning](https://lightning.ai/docs/pytorch/latest/) framework. 
The [torchtyping](https://github.com/patrick-kidger/torchtyping) library is used to annotate tensor types and shapes 
in function signatures. 
Unit test have been created to do initial sanity checks for the model. 

# Installation
To clone the repo and create a conda environment with the relevant libraries installed run:
```
git clone https://github.com/bmistry4/rims.git
cd rims
conda create --name rim-env --file requirements.txt
```

# Copy task 
Example models to run the copying task can be found in `jobs/copying.sh`.

# Architecture notes

**Rims** 
- Analogous to `rnn_models_wiki` in original code
- Deals with looping over layers (imagine a stacked LSTM) and looping over the timesteps.
- `layers:list[RimsCell]`

**RimsCell** 
- RimsCell ~ LSTMCell so the 'Rims' handle multiple rims in the cell. 
- Analogous to `BlocksCore` in original code
- Does the steps for: input attn, indep dyn, comm attn
- Contains a generic `Cell` class which is extended depending on the type of implementation you want. 
    - Currently 2 implementations 
        1) original paper way (BlockCell) which calls blockify, 
        2) our way (BatchedCell) which uses a batch * |RIMs| dimension and does BMM. 
        3) (was also a third option of a list of Cells which update in a for loop, but this would bee too expensive wrt time complexity)
        
        Both implementations should work on both GRU and LSTM

**Cell**
- Highest level abstraction for a container representing a collection of recurrent cells
- `Cell(input:Tensor, states:list/tuple) -> states:list/tuple`
- `state`: assume hidden sate is at index 0, and all states will apply the same type of masking as the hidden state

**Attention Mechnaism**
- Multi-headed Attention (MHA)
- Scaled dot prod attn (SDPA)
- Abstracts the sparse attention. Can be done at the SDPA level or at the MHA level. Or, use the `top_k=-1` to be the 
default (for dense MHA) and a value >0 to use sparse attention  

