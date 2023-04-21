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



gitRoot=os.path.join(os.environ['HOME'], "git", "bgnn")
simGitDir=os.path.join(gitRoot, "problems")
repo=git.Repo(gitRoot)
repoId=repo.head.object.hexsha

simDir=os.path.join("/system/user/mayr-data/BGNN/hopper/")
runDir=os.path.join(simDir, "runsi")



nrSimRunsStart=0
nrSimRuns=40
nrSimRunsStop=nrSimRunsStart+nrSimRuns




destPathBase=runDir

nokeepProbArr=[]
velXArr=[]
velYArr=[]
velZArr=[]
cutMainIndArr=[]
angleArr=[]
holeSizeArr=[]
translationTableArr=[]

for simInd in range(nrSimRunsStart, nrSimRunsStop):
  destPath=destPathBase+"/"+str(simInd)
  
  parFile=open(os.path.join(destPath, "nokeepProb.pckl"),"rb"); nokeep=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velX.pckl"),"rb"); velX=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velY.pckl"),"rb"); velY=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velZ.pckl"),"rb"); velZ=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cutMainInd.pckl"),"rb"); cutMainInd=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "angle.pckl"),"rb"); angle=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "holeSize.pckl"),"rb"); holeSize=pickle.load(parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "translationTable.pckl"),"rb"); translationTable=pickle.load(parFile); parFile.close()
  
  
  
  nokeepProbArr.append(nokeep)
  velXArr.append(velX)
  velYArr.append(velY)
  velZArr.append(velZ)
  cutMainIndArr.append(cutMainInd)
  angleArr.append(angle)
  holeSizeArr.append(holeSize)

parTab=pd.DataFrame({
"nokeepProb": nokeepProbArr,
"velX": velXArr,
"velY": velYArr,
"velZ": velZArr,
"cutMainInd": cutMainIndArr,
"angle": angleArr,
"holeSize": holeSizeArr,
})
parTab.to_csv(destPathBase+"/parameters.csv")

