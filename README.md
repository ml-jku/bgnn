# bgnn
## Introduction

This is the **code repository** for our paper *Boundary Graph Neural Networks for 3D Simulations*, which was presented in the technical program at the [Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23)](https://aaai-23.aaai.org/) in Washington, D.C.  

A **data repository** is available at [https://ml.jku.at/research/bgnn/download/](https://ml.jku.at/research/bgnn/download/).

Currently the conference proceedings from 2023 do not yet seem to be available. In the meantime, you can access our associated [arXiv manuscript](https://arxiv.org/abs/2106.11299), where you can also find the technical appendix.

BibTeX (arXiv manuscript):

>`@article{bib:Mayr2023,`\
>&nbsp;&nbsp;`title={{Boundary Graph Neural Networks for 3D Simulations}},`\
>&nbsp;&nbsp;`author={Mayr, Andreas and Lehner, Sebastian and Mayrhofer, Arno and Kloss, Christoph and Hochreiter, Sepp and Brandstetter, Johannes},`\
>&nbsp;&nbsp;`year={2023},`\
>&nbsp;&nbsp;`eprint={2106.11299},`\
>&nbsp;&nbsp;`archivePrefix={arXiv},`\
>&nbsp;&nbsp;`primaryClass={cs.LG}`\
>`}`  

## Simulation Data

General simulation settings used by the scripts in this repository (i.e., especially randomly chosen hyperparameters, seeds, etc.) are given by the file [data/bgnn.zip](data/bgnn.zip).

Simulation trajectories are obtained by the simulation tool [LIGGGHTS](https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/tree/9fb7f67592be9304afca9cb6840892b3b7d048d6).
LIGGGHTS is dependent on 2 software libraries:
- VTK (tested version: 8.2.0), available at [https://www.vtk.org/files/release/8.2/VTK-8.2.0.tar.gz](https://www.vtk.org/files/release/8.2/VTK-8.2.0.tar.gz)
- MPICH (tested version: 3.3), available at [https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz](https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz)

Further, CMake might be needed for compilation. We used version 3.7.2 (available at [http://www.cmake.org/files/v3.7/cmake-3.7.2.tar.gz](http://www.cmake.org/files/v3.7/cmake-3.7.2.tar.gz)).
`VTK_INC_USR` and `VTK_LIB_USR` need to be adjusted wrt. the VTK installation in `Makefile.user` before installing LIGGGHTS.

Most simple in creating the used simulation data is possibly following the pipeline described by the script [scripts/dataCreationPipeline.sh](scripts/dataCreationPipeline.sh). First of all, machines are initialized with an initial particle filling, which may then be modified by arbitrary operations. Afterwards the simulations of interest are run. Finally, data is converted to NumPy format and there is some precomputation in order to speed up training.

## BGNN Model Training and Inference

BGNN model training and inference for the hopper and the rotating drum are implemented in TensorFlow 2 and are available in the directory [bgnn/tf](bgnn/tf).
Training and inference code for the mixer are implemented in PyTorch and are available in the directory [bgnn/pytorch](bgnn/pytorch).

Most of the time the main entry points are the `processXY.py`-scripts with `X` being the problem name (hopper, (rotating) drum, mixer) and `Y` being a version.
There are several command-line parameters, which work in principle as follows: If one, wants to do inference using an already trained and saved model, or one wants to continue training a saved model, the name of the model directory needs to be specified for the argument `-saveName`. Otherwise, if a model should be trained from scratch, the argument should be kept empty. `-train` determines, whether the script should be run in train or in inference mode. `-problem` and `-experiment` determine an actual directory with ground-truth simulation trajectories. `-execRollout` determines, whether rollout evaluations should be executed  during training. `-evalSequence` and `-plotSequence` allow to specify which sequences should be evaluated during training or at the inference stage. Usually, the convention was to use sequences 0-29 for training, sequences 30-34 for hyperparameter selection, and sequences 35-39 for final testing/evaluation. The number of rollout-timesteps can be adjusted by `-evalTimesteps`; `-modulo`, `-vtkOutput`, and `-npOutput` allow to specify the frequency and whether files with the corresponding output formats should be created for the predicted rollout sequences.

`Y=1` denotes the script for the cohesive simulation setting for hopper and drum, while `Y=2` denotes the script corresponding to the non-cohesive setting. For hopper, `Y=0` corresponds to the simulation experiment in chapter *TApp. C.2*. For both, hopper, and rotating drum `Y=i` corresponds to simulation experiments in the manuscript [Learning 3D Granular Flow Simulations](https://arxiv.org/abs/2105.01636). `Y=3` (for hoper only) and `Y=4` (for hoper and rotating drum) corresponds to out-of-distribution experiments. For the mixer we had only one setting, which corresponds to `Y=1`.

The `modelXY.py`-scripts form together with the `blocks.py`-script the actual implementation of the core graph network functionalities.
The functionality was derived from the [Graph Nets](https://www.deepmind.com/open-source/graph-nets) library of Google DeepMind.
We applied some adaptions to their code. The actual graph network model architecture is also determined by `confXY.py`-configuration scripts.

Rollout functionality is implemented by `rollout*.py`-scripts. Node and edge feature computation is implemented in `featureOptions1.py` and `featureOptions2.py`.
`util.py` implements basic utility functions, e.g., computing feature statistics for normalization purposes. Triangle-Point distance computation is implemented by `wallInfo_tf.py` and `wallInfo_torch.py`.


\
\
\
Update: April, 21st, 2023\
We are now providing the whole pipeline for producing data as well as training and prediction code for BGNNs.

Last Update: May, 15th, 2023
