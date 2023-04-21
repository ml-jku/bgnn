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



#Script Tools
cutToolsScript=os.path.join(simGitDir, "cutTools", "tools.py")
randomCutToolsScript=os.path.join(simGitDir, "cutTools", "randomTools.py")
writeCutStateScript=os.path.join(simGitDir, "cutTools", "writeState.py")

exec(open(cutToolsScript).read(), globals())
exec(open(randomCutToolsScript).read(), globals())
exec(open(writeCutStateScript).read(), globals())



nrSimRunsStart=100
nrSimRunsStop=130



destPathBase=runDir

randFnct=[rand0, rand1, rand2, rand3, rand4, rand5, rand6, rand7, rand8, rand9]
for simInd in range(nrSimRunsStart, nrSimRunsStop):
  destPath=os.path.join(destPathBase, str(simInd))
  
  
  
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
  parFile=open(os.path.join(destPath, "translationTable.pckl"),"rb"); translationTable=pickle.load(parFile); parFile.close()
  
  
  
  simFilesParticlesPrefix="init_particles_"
  simFilesParticles=glob.glob(os.path.join(destPath, "post", simFilesParticlesPrefix+"*.vtp"))
  savedTimesParticles=np.sort([int(x.split("_")[-1].split(".vtp")[0]) for x in simFilesParticles])
  strTemplParticles=os.path.join(destPath, "post", simFilesParticlesPrefix+"{0}.vtp")
  strTemplsParticles=[strTemplParticles.format(x) for x in savedTimesParticles]
  
  readFile=strTemplsParticles[-1]
  reader=vtk.vtkXMLPolyDataReader()
  reader.SetFileName(readFile)
  reader.Update()
  simData=reader.GetOutputAsDataSet(0)
  
  x_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetPoints().GetData())
  id_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("id"))
  type_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("type"))
  i_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("i"))
  v_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("v"))
  f_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("f"))
  radius_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("radius"))
  omega_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("omega"))
  tq_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("tq"))
  density_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("density"))
  
  
  #myrand=np.random.randint(100000)+int(time.time()*1000000)%100000
  #np.save(os.path.join(destPath, "myrseed"), myrand)
  myrand=np.load(os.path.join(destPath, "myrseed.npy"))
  
  np.random.seed(myrand)
  positionData=(x_Data.min(0)[0], x_Data.min(0)[1], 0.0, x_Data.max(0)[0]-x_Data.min(0)[0], x_Data.max(0)[1]-x_Data.min(0)[1], x_Data.max(0)[2]-x_Data.min(0)[2])
  cut, par=randFnct[cutMainInd](x_Data, positionData, minPoints=1000, maxPoints=8000, nrTrials=100)
  
  #parFile=open(os.path.join(destPath, "cut.pckl"),"wb")
  #pickle.dump(par, parFile)
  #parFile.close()
  
  #np.random.seed(int(time.time()*1000000)%1000)
  
  cut[np.random.permutation(np.arange(len(cut)))[int(len(cut)*nokeep):]]=False
  cut[:]=True
  
  
  cutAway=np.where(cut)[0]
  cutAway=np.random.permutation(cutAway)
  if simInd%5==0:
    cut[cutAway[0:math.ceil(len(cutAway)*0.1)]]=False
  elif simInd%5==1:
    cut[cutAway[0:math.ceil(len(cutAway)*0.2)]]=False
  elif simInd%5==2:
    cut[cutAway[0:math.ceil(len(cutAway)*0.3)]]=False
  elif simInd%5==3:
    cut[cutAway[0:math.ceil(len(cutAway)*0.4)]]=False
  elif simInd%5==4:
    cut[cutAway[0:math.ceil(len(cutAway)*0.5)]]=False
  if simInd<105:
    cut[:]=True
  
  
  v_Data=v_Data+np.array([velX, velY, velZ]).reshape(1,3)
  
  id_DataNew=np.arange(np.sum(cut))+1
  #id_DataNew=id_Data
  x_DataNew=x_Data[cut]
  v_DataNew=v_Data[cut]
  omegaDataNew=omega_Data[cut]
  radius_DataNew=radius_Data[cut]
  type_DataNew=type_Data[cut]
  density_DataNew=density_Data[cut]
  
  nrAtomTypes=3
  regionX0=-0.6
  regionX1=0.6
  regionY0=-0.6
  regionY1=0.6
  regionZ0=-0.6
  regionZ1=0.6
  
  #print(destPath+"/init.data")
  writeState(destPath+"/init.data", regionX0, regionX1, regionY0, regionY1, regionZ0, regionZ1, id_DataNew, x_DataNew, v_DataNew, omegaDataNew, radius_DataNew, type_DataNew, density_DataNew, nrAtomTypes)
