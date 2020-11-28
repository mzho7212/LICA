# LICA 1-Step Traffic Junction Env

This directory holds a simple PyTorch implementation of the 1-Step Traffic Junction component study (Section 4.3.1) in the paper. Note that:

1. Due to the simplicity of the environment, there's only one possible state, so we don't actually need to formulate a state vector. (We do however need to include an agent id to distinguish the agents.)

2. As only one possible state exists, the hypernetwork architecture can be reduced to MLP since the state-to-weights mapping will always have the same state input. 

3. Training for 20k random initializations takes roughly 25 minutes with 4 cpu cores. 

## Training & Ploting
Run `python3 plot.py <number of random initializations>`
