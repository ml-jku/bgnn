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
import shutil
import stl
import primesieve
import more_itertools
import copy



gitRoot=os.path.join(os.environ['HOME'], "git", "bgnn")
simGitDir=os.path.join(gitRoot, "problems")
repo=git.Repo(gitRoot)
repoId=repo.head.object.hexsha

simDir=os.path.join("/system/user/mayr-data/BGNN/mixer/")
runDir=os.path.join(simDir, "runs1")



wallInfoScript=os.path.join(simGitDir, "code", "general", "wallInfo_tf.py")



nrSimRunsStart=0
nrSimRuns=40
nrSimRunsStop=nrSimRunsStart+nrSimRuns



destPathBase=runDir

xParticleDataList=[]
#xSceneDataList=[]
maxRadiusList=[]

for simInd in range(0, nrSimRunsStop):
  print(simInd)
  destPath=destPathBase+"/"+str(simInd)
  
  xParticleData=np.load(os.path.join(destPath, "xData.npy"), mmap_mode="r")
  xParticleDataList.append(xParticleData)
  
  #xSceneData=np.load(os.path.join(destPath, "triangleCoordDataNew.npy"), mmap_mode="r")
  #xSceneDataList.append(xSceneData)
  
  parFile=open(os.path.join(destPath, "radMain.pckl"),"rb"); radMain=pickle.load(parFile); parFile.close()
  maxRadiusList.append(max([np.array(x).max() for x in radMain if x is not None]))

destPath=os.path.join(runDir, "0")
destMeshPath=os.path.join(destPath, 'meshes')
blade1=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Blade1.stl'))
blade4=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Blade4.stl'))
mixer1=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Mixer1.stl'))
mixer4=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Mixer4.stl'))
mixingDrum=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'mixingDrum.stl'))
shaft=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Shaft.stl'))
parFile=open(os.path.join(destPath, "rpMain.pckl"),"rb"); rpMain=pickle.load(parFile); parFile.close()

computeTimestep=1.0e-5
statusTimestep=1e-3
dumpTimeStep=1e-3

#angle=(360.0/((rpMain/1e-05)/100.))
angle=(360.0/((rpMain/computeTimestep)/(dumpTimeStep/computeTimestep)))



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
  #xSceneData=xSceneDataList[i]
  xParticleData=xParticleDataList[i]
  maxRadius=maxRadiusList[i]
  neighborCutoff=4.0*maxRadius
  maxParticleNoiseData=[]
  maxParticleCoordNoiseDataPos=[]
  maxParticleCoordNoiseDataNeg=[]
  for j in range(0, len(xParticleDataList[i])):
    print(f"{i}/{j}", end="\r")
    with tf.device('/device:GPU:0'):
      points=tf.identity(np.float32(xParticleData[j,:]))
      myiter=j
      rotMatIter=np.array([[1.0, 0.0, 0.0], [0.0, np.cos(((angle*myiter)*math.pi)/180), -np.sin(((angle*myiter)*math.pi)/180)], [0.0, np.sin(((angle*myiter)*math.pi)/180), np.cos(((angle*myiter)*math.pi)/180)]])
      rotBlade1=np.matmul(blade1.vectors, rotMatIter.T)
      rotBlade4=np.matmul(blade4.vectors, rotMatIter.T)
      rotMixer1=np.matmul(mixer1.vectors, rotMatIter.T)
      rotMixer4=np.matmul(mixer4.vectors, rotMatIter.T)
      rotShaft=np.matmul(shaft.vectors, rotMatIter.T)
      rotMixingDrum=mixingDrum.vectors
      rotMesh=np.vstack([rotShaft, rotBlade1, rotBlade4, rotMixer1, rotMixer4, rotMixingDrum])
      xSceneData=rotMesh
      #tcoord=tf.identity(np.float32(xSceneData[j]))
      tcoord=tf.identity(np.float32(xSceneData))
    exec(open(wallInfoScript).read(), globals())
    maxParticleNoiseData.append(tf.reduce_min(dists,1).numpy())
    v1=tf.math.unsorted_segment_min(tf.gather_nd(tf.where(diffVec>0.0, diffVec, neighborCutoff), takeDist), takeDist[:,0],dists.shape[0])
    v1=tf.where(v1>10**30, neighborCutoff, v1)
    maxParticleCoordNoiseDataPos.append(v1)
    v2=tf.math.unsorted_segment_max(tf.gather_nd(tf.where(diffVec<0.0, diffVec, -neighborCutoff), takeDist), takeDist[:,0],dists.shape[0])
    v2=tf.where(v2<-10**30, -neighborCutoff, v2)
    maxParticleCoordNoiseDataNeg.append(v2)
  
  maxParticleNoiseData=np.array(maxParticleNoiseData)
  maxParticleCoordNoiseDataPos=np.array(maxParticleCoordNoiseDataPos)
  maxParticleCoordNoiseDataNeg=np.array(maxParticleCoordNoiseDataNeg)
  np.save(os.path.join(destPath, "particleWallDistV5"), maxParticleNoiseData)
  np.save(os.path.join(destPath, "particleWallDiffPosV5"), maxParticleCoordNoiseDataPos)
  np.save(os.path.join(destPath, "particleWallDiffNegV5"), maxParticleCoordNoiseDataNeg)
print()
