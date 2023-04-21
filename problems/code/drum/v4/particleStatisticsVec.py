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
os.environ['CUDA_VISIBLE_DEVICES']=''
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

simDir=os.path.join("/system/user/mayr-data/BGNN/drum/")
runDir=os.path.join(simDir, "runs4")



nrSimRunsStart=0
nrSimRuns=40
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



class KahanSumExtended:
  def __init__(self):
    self.c1=np.zeros(3, dtype=np.float64)
    self.c2=np.zeros(3, dtype=np.float64)
    self.c3=np.zeros(3, dtype=np.float64)

  def add(self, nr):
    c1=self.c1+nr
    c2=self.c2+c1
    self.c1=c1-(c2-self.c2)
    c3=self.c3+c2
    self.c2=c2-(c3-self.c3)
    self.c3=c3
  
  def readout(self):
    return self.c1+self.c2+self.c3



import pymp
with pymp.Parallel(40) as parproc:
 for i in parproc.range(nrSimRunsStart, nrSimRunsStop):
  print(i)
  destPath=destPathBase+"/"+str(i)
  vel=(xParticleDataList[i][1:]-xParticleDataList[i][:-1])
  acc=(vel[1:]-vel[:-1])
  vel=vel.reshape(-1,3)
  acc=acc.reshape(-1,3)
  overallVelSum=KahanSumExtended()
  overallVelSum2=KahanSumExtended()
  overallAccSum=KahanSumExtended()
  overallAccSum2=KahanSumExtended()
  for j in range(0, vel.shape[0]):
    overallVelSum.add(vel[j,:])
    overallVelSum2.add(vel[j,:]**2)
  for j in range(0, acc.shape[0]):
    overallAccSum.add(acc[j,:])
    overallAccSum2.add(acc[j,:]**2)
  overallVelSum=overallVelSum.readout()
  overallVelSum2=overallVelSum2.readout()
  overallAccSum=overallAccSum.readout()
  overallAccSum2=overallAccSum2.readout()
  destPath=destPathBase+"/"+str(i)
  np.save(os.path.join(destPath, "overallVelSum"), overallVelSum)
  np.save(os.path.join(destPath, "overallVelSum2"), overallVelSum2)
  np.save(os.path.join(destPath, "overallAccSum"), overallAccSum)
  np.save(os.path.join(destPath, "overallAccSum2"), overallAccSum2)
  np.save(os.path.join(destPath, "VelElems"), np.int64(vel.shape[0]))
  np.save(os.path.join(destPath, "AccElems"), np.int64(acc.shape[0]))
print()
