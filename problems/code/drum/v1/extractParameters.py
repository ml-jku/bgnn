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

simDir=os.path.join("/system/user/mayr-data/BGNN/drum/")
runDir=os.path.join(simDir, "runs1")




nrSimRunsStart=0
nrSimRuns=40
nrSimRunsStop=nrSimRunsStart+nrSimRuns




destPathBase=runDir

nokeepProbArr=[]
velXArr=[]
velYArr=[]
velZArr=[]
cutMainIndArr=[]
angleInitArr=[]
angleMainArr=[]
rpMainArr=[]
ymInitArr=[]
ymMainArr=[]
prInitArr=[]
prMainArr=[]
crInitArr=[]
crMainArr=[]
cfInitArr=[]
cfMainArr=[]
crfInitArr=[]
crfMainArr=[]
cedInitArr=[]
cedMainArr=[]
densityInitArr=[]
densityMainArr=[]
radInitArr=[]
radMainArr=[]
insRateInitArr=[]
translationTableArr=[]

for simInd in range(nrSimRunsStart, nrSimRunsStop):
  destPath=destPathBase+"/"+str(simInd)
  
  parFile=open(os.path.join(destPath, "nokeepProb.pckl"),"rb"); nokeep=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velX.pckl"),"rb"); velX=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velY.pckl"),"rb"); velY=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velZ.pckl"),"rb"); velZ=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cutMainInd.pckl"),"rb"); cutMainInd=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "angleInit.pckl"),"rb"); angleInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "angleMain.pckl"),"rb"); angleMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "rpMain.pckl"),"rb"); rpMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "ymInit.pckl"),"rb"); ymInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "ymMain.pckl"),"rb"); ymMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "prInit.pckl"),"rb"); prInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "prMain.pckl"),"rb"); prMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crInit.pckl"),"rb"); crInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crMain.pckl"),"rb"); crMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cfInit.pckl"),"rb"); cfInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cfMain.pckl"),"rb"); cfMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crfInit.pckl"),"rb"); crfInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crfMain.pckl"),"rb"); crfMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cedInit.pckl"),"rb"); cedInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cedMain.pckl"),"rb"); cedMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "densityInit.pckl"),"rb"); densityInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "densityMain.pckl"),"rb"); densityMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "radInit.pckl"),"rb"); radInit=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "radMain.pckl"),"rb"); radMain=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "insRateInit.pckl"),"rb"); insRateInit=pickle.load(parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "translationTable.pckl"),"rb"); translationTable=pickle.load(parFile); parFile.close()  
  
  
  
  nokeepProbArr.append(nokeep)
  velXArr.append(velX)
  velYArr.append(velY)
  velZArr.append(velZ)
  cutMainIndArr.append(cutMainInd)
  angleInitArr.append(angleInit)
  angleMainArr.append(angleMain)
  rpMainArr.append(rpMain)
  ymInitArr.append(ymInit)
  ymMainArr.append(ymMain)
  prInitArr.append(prInit)
  prMainArr.append(prMain)
  crInitArr.append(crInit)
  crMainArr.append(crMain)
  cfInitArr.append(cfInit)
  cfMainArr.append(cfMain)
  crfInitArr.append(crfInit)
  crfMainArr.append(crfMain)
  cedInitArr.append(cedInit)
  cedMainArr.append(cedMain)
  densityInitArr.append(densityInit)
  densityMainArr.append(densityMain)
  radInitArr.append(radInit)
  radMainArr.append(radMain)
  insRateInitArr.append(insRateInit)


parTab=pd.DataFrame({
"nokeepProb": nokeepProbArr,
"velX": velXArr,
"velY": velYArr,
"velZ": velZArr,
"cutMainInd": cutMainIndArr,
"angleInit": angleInitArr,
"angleMain": angleMainArr,
"rpMain": rpMainArr,
"ymInit": ymInitArr,
"ymMain": ymMainArr,
"prInit": prInitArr,
"prMain": prMainArr,
"crInit": crInitArr,
"crMain": crMainArr,
"cfInit": cfInitArr,
"cfMain": cfMainArr,
"crfInit": crfInitArr,
"crfMain": crfMainArr,
"cedInit": cedInitArr,
"cedMain": cedMainArr,
"densityInit": densityInitArr,
"densityMain": densityMainArr,
"radInit": radInitArr,
"radMain": radMainArr,
"insRateInit": insRateInitArr
})
parTab.to_csv(destPathBase+"/parameters.csv")

