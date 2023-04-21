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

import pymp
with pymp.Parallel(40) as parproc:
 for simInd in parproc.range(nrSimRunsStart, nrSimRunsStop):
  print(simInd)
  destPath=destPathBase+"/"+str(simInd)

  simFilesWallsPrefix="main_walls_"
  simFilesWalls=glob.glob(os.path.join(destPath, "post", simFilesWallsPrefix+"*.vtp"))
  savedTimesWalls=np.sort([int(x.split("_")[-1].split(".vtp")[0]) for x in simFilesWalls])
  strTemplWalls=os.path.join(destPath, "post", simFilesWallsPrefix+"{0}.vtp")
  strTemplsWalls=[strTemplWalls.format(x) for x in savedTimesWalls]
  
  triangleCoord_DataArr=[]
  normalStress_DataArr=[]
  shearStress_DataArr=[]
  
  for j in range(0, len(strTemplsWalls)):
    readFile=strTemplsWalls[j]
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(readFile)
    reader.Update()
    wallData=reader.GetOutputAsDataSet(0)
    
    triangePoint_Data=vtk.util.numpy_support.vtk_to_numpy(wallData.GetPoints().GetData())
    
    triangeInd_Data=vtk.util.numpy_support.vtk_to_numpy(wallData.GetPolys().GetData())
    assert(len(triangeInd_Data)//4==wallData.GetPolys().GetData().GetNumberOfTuples()//4)
    assert(len(triangeInd_Data)//4==wallData.GetPolys().GetNumberOfCells())
    
    triangleInd=[]
    start=0
    while start<len(triangeInd_Data):
      end=start+triangeInd_Data[start]+1
      start=start+1
      triangleInd.append(triangeInd_Data[start:end])
      start=end
    triangleInd=np.array(triangleInd)
    
    triangleCoords=[]
    for tnr in range(0, len(triangleInd)):
      pA=triangePoint_Data[triangleInd[tnr][0]]
      pB=triangePoint_Data[triangleInd[tnr][1]]
      pC=triangePoint_Data[triangleInd[tnr][2]]
      triangleCoords.append([pA, pB, pC])
    triangleCoord_Data=np.array(triangleCoords) #(triangleNr [0..nrTriangles-1], trainglePointNr [0..2], trianglePointCoord [0..2])
    normalStress_Data=vtk.util.numpy_support.vtk_to_numpy(wallData.GetAttributes(1).GetScalars("normal_stress_average"))
    shearStress_Data=vtk.util.numpy_support.vtk_to_numpy(wallData.GetAttributes(1).GetScalars("shear_stress_average"))
    
    triangleCoord_DataArr.append(triangleCoord_Data)
    normalStress_DataArr.append(normalStress_Data)
    shearStress_DataArr.append(shearStress_Data)
  
  triangleCoordData=np.array(triangleCoord_DataArr)
  normalStressData=np.array(normalStress_DataArr)
  shearStressData=np.array(shearStress_DataArr)
  
  np.save(os.path.join(destPath, "triangleCoordDataNew"), triangleCoordData)
  np.save(os.path.join(destPath, "normalStressDataNew"), normalStressData)
  np.save(os.path.join(destPath, "shearStressDataNew"), shearStressData)
