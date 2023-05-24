# bgnn
## Introduction

This is the **code repository** for our paper *Boundary Graph Neural Networks for 3D Simulations*, which was presented in the technical program at the [Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23)](https://aaai-23.aaai.org/) in Washington, D.C.  

A **data repository** is available at [https://ml.jku.at/research/bgnn/download/](https://ml.jku.at/research/bgnn/download/).

Currently the conference proceedings from 2023 do not yet seem to be available. In the meantime, you can access our associated [arXiv manuscript](https://arxiv.org/abs/2106.11299), where you can also find the technical appendix.

BibTeX (arXiv manuscript):
````
@article{bib:Mayr2023,
      title={Boundary Graph Neural Networks for 3D Simulations}, 
      author={Andreas Mayr and Sebastian Lehner and Arno Mayrhofer and Christoph Kloss and Sepp Hochreiter and Johannes Brandstetter},
      year={2023},
      eprint={2106.11299},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```` 

There is also a [Blog post](https://ml-jku.github.io/bgnn/) for a quick introduction to the paper.

## Preliminary Note

Please consider, to adjust the base directory names in the individual files according to your own environment (e.g., especially search for the strings `-data/BGNN/` and `-data/BGNNRuns/` and adjust the base directory names). Further, it is assumed, that this code repository is at `$HOME/git/bgnn`. For analysis scripts, a few further directories in `$HOME` are used, such as `bgnnPlots` or `bgnnInfo`. \
Basically, in the scripts it is assumed, that there is somewhere a `BGNN` and a `BGNNRuns` directory, where `BGNN` serves as a directory containing (ground truth) simulation data, while `BGNNRuns` serves as a working directory for the scripts in this repository. Please consider the chapter on [Comments on the Data Repository](#comments-on-the-data-repository) for more information how `BGNN` and `BGNNRuns` are expected to be built up. A possible way to start with building up the `BGNN` directory is either the extraction of [data/bgnn.zip](data/bgnn.zip) (contains only setting parameters, etc.) or the extraction of [BGNN.zip](https://ml.jku.at/research/bgnn/download/BGNN.zip) (also contains results of initialization runs). \
The used python version and a list of used packages are given by [pythonVersion.txt](pythonVersion.txt) and [pythonPackageVersions.txt](pythonPackageVersions.txt) respectively.

## Simulation Data

General simulation settings used by the scripts in this repository (i.e., especially randomly chosen hyperparameters, seeds, etc.) are given by the file [data/bgnn.zip](data/bgnn.zip).

Simulation trajectories are obtained by the simulation tool [LIGGGHTS](https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/tree/9fb7f67592be9304afca9cb6840892b3b7d048d6).
LIGGGHTS is dependent on 2 software libraries:
- VTK (tested version: 8.2.0), available at [https://www.vtk.org/files/release/8.2/VTK-8.2.0.tar.gz](https://www.vtk.org/files/release/8.2/VTK-8.2.0.tar.gz)
- MPICH (tested version: 3.3), available at [https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz](https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz)

Further, CMake might be needed for compilation. We used version 3.7.2 (available at [http://www.cmake.org/files/v3.7/cmake-3.7.2.tar.gz](http://www.cmake.org/files/v3.7/cmake-3.7.2.tar.gz)).
`VTK_INC_USR` and `VTK_LIB_USR` need to be adjusted wrt. the VTK installation in `Makefile.user` before installing LIGGGHTS.

Most simple in creating the used simulation data is possibly following the pipeline described by the script [scripts/dataCreationPipeline.sh](scripts/dataCreationPipeline.sh). First of all, machines are initialized with an initial particle filling (set and sample initial parameters by `createInit` followed by `execInit`), which may then be modified by arbitrary operations (`createMainCut`). Afterwards the simulations of interest are run (`execMain`). Finally, data is converted to NumPy format (`extractParticles`, `extractWalls`) and there is some precomputation in order to speed up training (`particleStatisticsLen`, `particleStatisticsVec`, `particleWallDistances`).

The code scripts can be found in [problems](problems). Especially, we considered a hopper, a rotating drum, and a mixer geometry. Detailed scripts for each step in the pipeline are available in the subdirectories of [problems/code](problems/code). Consider, that for some problems, uncertainty computations are included (cohesive and non-cohesive hopper and rotating drum) and therefore there are a few more steps to create all data for these problems. Further, for out-of-distribution experiments, there are several `createInit*`-scripts. The implementation of arbitrary operations applied to initial particle fillings are available at [problems/cutTools](problems/cutTools). Raw LIGGGHTS template scripts can be found at [problems/templates](problems/templates).

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
`util.py` implements basic utility functions, e.g., computing feature statistics for normalization purposes. Triangle-Point distance computation is implemented by `wallInfo_tf.py` and `wallInfo_torch.py`. `saveModel.py` and `loadModel.py` are used to store BGNN models to disk and load them from the disk.

Code for the simulation experiment in chapter *TApp. C.2* can be found in [bgnn/tf/fexp](bgnn/tf/fexp), while code for ablation experiments (chapter *TApp. D.5*) can be found in [bgnn/tf/ablations](bgnn/tf/ablations). The structure of the scripts follows in principle those in [bgnn/tf](bgnn/tf) with some adaptions.

## Analysis and Utility Scripts

Scripts to reproduce curves of analysis plots in the paper are provided at [scripts/visualizations](scripts/visualizations). `hopper_initial.py` and `drum_initial.py` correspond to plots in [Learning 3D Granular Flow Simulations](https://arxiv.org/abs/2105.01636).

Code how statistical tests in *TApp. C.2* and *TApp. D.4* were applied can be found in `testC2.py` and `testD4.py` in [scripts](scripts) respectively. The numbers in *Table 1* were obtained by scripts in [scripts/grw](scripts/grw).

Further, some general utility scripts, such as converting NumPy data to VTK and vice versa are provided in  [scripts](scripts).

To visualize particle trajectories given by VTK files, [ParaView](https://www.paraview.org/) can be used.

## Comments on the Data Repository

Availability:  [https://ml.jku.at/research/bgnn/download/](https://ml.jku.at/research/bgnn/download/)

The repository consists of 4 ZIP files:
- [BGNN.zip](https://ml.jku.at/research/bgnn/download/BGNN.zip): contains parameters to reproduce simulations + run of initial filling and random operations (`createMainCut`); main simulations runs (beginning with `execMain`), conversion, and precomputation steps are still necessary (see [Simulation Data](#simulation-data)) as the file for download would otherwise get quite large

- [models.zip](https://ml.jku.at/research/bgnn/download/models.zip): contains saved BGNN models
- [trajectories.zip](https://ml.jku.at/research/bgnn/download/trajectories.zip): contains trajectory rollouts used in the publication
- [evaluations.zip](https://ml.jku.at/research/bgnn/download/evaluations.zip): contains evaluation results, which were further used in the publication (not computed for all trajectories)

Ideally the content of `BGNN.zip` is extracted to a directory `$BGNN_BASEDIR/BGNN`, while the content of the other files is extracted into a directory `$BGNN_BASEDIR/BGNNRuns`, such that `$BGNN_BASEDIR/BGNNRuns` consists of subdirectories containing `models`, `trajectories`, and `evaluations`. Further,  (empty) subdirectories with the names `info` and `predictions` should be created in `$BGNN_BASEDIR/BGNNRuns`. `$BGNN_BASEDIR/BGNN` consists of subdirectories `hopper`, `drum`, and `mixer`.


\
\
Last Update: May, 24th, 2023
