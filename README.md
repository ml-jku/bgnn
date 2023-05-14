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
\
\
\
Update: April, 21st, 2023\
We are now providing the whole pipeline for producing data as well as training and prediction code for BGNNs.

Last Update: May, 15th, 2023
