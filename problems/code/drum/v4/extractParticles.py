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
runDir=os.path.join(simDir, "runs4")



nrSimRunsStart=0
nrSimRuns=40
nrSimRunsStop=nrSimRunsStart+nrSimRuns



destPathBase=runDir

import pymp
with pymp.Parallel(40) as parproc:
 for simInd in parproc.range(nrSimRunsStart, nrSimRunsStop):
  print(simInd)

  destPath=destPathBase+"/"+str(simInd)
  
  simFilesParticlesPrefix="main_particles_"
  simFilesParticles=glob.glob(os.path.join(destPath, "post", simFilesParticlesPrefix+"*.vtp"))
  savedTimesParticles=np.sort([int(x.split("_")[-1].split(".vtp")[0]) for x in simFilesParticles])
  strTemplParticles=os.path.join(destPath, "post", simFilesParticlesPrefix+"{0}.vtp")
  strTemplsParticles=[strTemplParticles.format(x) for x in savedTimesParticles]
  
  x_DataArr=[]
  id_DataArr=[]
  type_DataArr=[]
  i_DataArr=[]
  v_DataArr=[]
  f_DataArr=[]
  radius_DataArr=[]
  omega_DataArr=[]
  tq_DataArr=[]
  density_DataArr=[]
  
  for j in range(0, len(strTemplsParticles)):
    readFile=strTemplsParticles[j]
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(readFile)
    reader.Update()
    simData=reader.GetOutputAsDataSet(0)

    x_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetPoints().GetData())
    id_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("id"))
    type_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("type"))
    radius_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("radius"))
    density_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("density"))
    # i_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("i"))
    # v_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("v"))
    # f_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("f"))
    # omega_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("omega"))
    # tq_Data=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetVectors("tq"))
    
    sortOrder=np.argsort(id_Data)
    
    x_DataArr.append(x_Data[sortOrder,:])
    id_DataArr.append(id_Data[sortOrder])
    type_DataArr.append(type_Data[sortOrder])
    radius_DataArr.append(radius_Data[sortOrder])
    density_DataArr.append(density_Data[sortOrder])
    # i_DataArr.append(i_Data[sortOrder,:])
    # v_DataArr.append(v_Data[sortOrder,:])
    # f_DataArr.append(f_Data[sortOrder,:])
    # omega_DataArr.append(omega_Data[sortOrder,:])
    # tq_DataArr.append(tq_Data[sortOrder,:])
  
  xData=np.array(x_DataArr)
  idData=np.array(id_DataArr)
  typeData=np.array(type_DataArr)
  radiusData=np.array(radius_DataArr)
  densityData=np.array(density_DataArr)
  # iData=np.array(i_DataArr)
  # vData=np.array(v_DataArr)
  # fData=np.array(f_DataArr)
  # omegaData=np.array(omega_DataArr)
  # tqData=np.array(tq_DataArr)
  
  np.save(os.path.join(destPath, "xData"), xData)
  np.save(os.path.join(destPath, "idData"), idData)
  np.save(os.path.join(destPath, "typeData"), typeData)
  np.save(os.path.join(destPath, "radiusData"), radiusData)
  np.save(os.path.join(destPath, "densityData"), densityData)
  # #np.save(os.path.join(destPath, "iData"), iData)
  # np.save(os.path.join(destPath, "vData"), vData)
  # np.save(os.path.join(destPath, "fData"), fData)
  # #np.save(os.path.join(destPath, "radiusData"), radiusData)
  # np.save(os.path.join(destPath, "omegaData"), omegaData)
  # np.save(os.path.join(destPath, "tqData"), tqData)