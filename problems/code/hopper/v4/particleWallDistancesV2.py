#Source codes of the publication "Boundary Graph Neural Networks for 3D Simulations"
#Copyright (C) 2023  Andreas Mayr

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#The license file for GNU General Public License v3.0 is available here:
#https://github.com/ml-jku/bgnn/blob/master/licenses/own/LICENSE_GPL3

import numpy as np
import pandas as pd
#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format
import scipy
import math
import glob
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import sys
import importlib
import h5py
import vtk
import vtk.util
import vtk.util.numpy_support
import pickle
import scipy
import scipy.stats
import git
import pathlib
import logging
import itertools



gitRoot=os.path.join(os.environ['HOME'], "git", "bgnn")
simGitDir=os.path.join(gitRoot, "problems")
repo=git.Repo(gitRoot)
repoId=repo.head.object.hexsha

simDir=os.path.join("/system/user/mayr-data/BGNN/hopper/")
runDir=os.path.join(simDir, "runs4")



wallInfoScript=os.path.join(simGitDir, "code", "general", "wallInfo_tf.py")



nrSimRunsStart=0
nrSimRuns=70
nrSimRunsStop=nrSimRunsStart+nrSimRuns



destPathBase=runDir

xParticleDataList=[]
xSceneDataList=[]

for simInd in range(0, nrSimRunsStop):
  print(simInd)
  destPath=destPathBase+"/"+str(simInd)
  
  xParticleData=np.load(os.path.join(destPath, "xData.npy"), mmap_mode="r")
  xParticleDataList.append(xParticleData)
  
  xSceneData=np.load(os.path.join(destPath, "triangleCoordDataNew.npy"), mmap_mode="r")
  xSceneDataList.append(xSceneData)

gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
availGPUs={0:0} #logical number should be consecutive, physical numbers correspond to physcial GPU Ids
nrAvailGPUs=len(availGPUs)
logGPU=availGPUs[gpu]
nrSimRunsStart=logGPU*int(np.ceil(float(nrSimRuns)/float(nrAvailGPUs)))
nrSimRunsStop=min((logGPU+1)*int(np.ceil(float(nrSimRuns)/float(nrAvailGPUs))), nrSimRuns)
neighborCutoff=0.0

import pymp
with pymp.Parallel(10) as parproc:
 for i in parproc.range(nrSimRunsStart, nrSimRunsStop):
  import tensorflow as tf
  tf.get_logger().setLevel(logging.ERROR)
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(1)
  
  print(i)
  destPath=destPathBase+"/"+str(i)
  xSceneData=xSceneDataList[i]
  xParticleData=xParticleDataList[i]
  maxParticleNoiseData=[]
  maxParticleCoordNoiseDataPos=[]
  maxParticleCoordNoiseDataNeg=[]
  for j in range(0, len(xParticleDataList[i])):
    print(f"{i}/{j}", end="\r")
    with tf.device('/device:GPU:0'):
      points=tf.identity(np.float32(xParticleData[j,:]))
      tcoord=tf.identity(np.float32(xSceneData[j]))
    exec(open(wallInfoScript).read(), globals())
    maxParticleNoiseData.append(tf.reduce_min(dists,1).numpy())
    maxParticleCoordNoiseDataPos.append(tf.reduce_min(tf.where(diffVec>0.0, diffVec, np.inf), 1))
    maxParticleCoordNoiseDataNeg.append(tf.reduce_max(tf.where(diffVec<0.0, diffVec, -np.inf), 1))
    
  maxParticleNoiseData=np.array(maxParticleNoiseData)
  maxParticleCoordNoiseDataPos=np.array(maxParticleCoordNoiseDataPos)
  maxParticleCoordNoiseDataNeg=np.array(maxParticleCoordNoiseDataNeg)
  np.save(os.path.join(destPath, "particleWallDistV2"), maxParticleNoiseData)
  np.save(os.path.join(destPath, "particleWallDiffPosV2"), maxParticleCoordNoiseDataPos)
  np.save(os.path.join(destPath, "particleWallDiffNegV2"), maxParticleCoordNoiseDataNeg)
print()
