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
np.set_printoptions(suppress=False)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format
import scipy
import math
import collections
import glob
import os
import shutil
#os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS'] = "4"
import sys
import copy
import importlib
import h5py
import vtk
import vtk.util
import vtk.util.numpy_support
import pickle
import scipy
import scipy.stats
import sklearn
import sklearn.neighbors
import git
import pathlib
import logging
import itertools
import termios
import fcntl
import select
import time
import datetime
import IPython
import socket
import pprint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import json
import ot
import stl



rp=collections.OrderedDict()
rp["problem"]="mixer"
rp["experiment"]="runs1"
rp["stepSize"]=1
rp["nrPastVelocities"]=2

saveDir="/system/user/mayr-data/BGNNRuns/models/"+rp["problem"]+"/"+rp["experiment"]+"/"
predictTimesteps=5000
modulo=10
saveName="wall"+str(modulo)

rolloutFilePrefix=os.path.join("/system/user/mayr-data/BGNNRuns/predictions", rp["problem"], rp["experiment"], saveName)
if not os.path.exists(os.path.join(rolloutFilePrefix)):
  os.makedirs(rolloutFilePrefix)

destPathBase="/system/user/mayr-data/BGNN/"+rp["problem"]+"/"+rp["experiment"]
destMeshPath=os.path.join(destPathBase, '0', 'meshes')
blade1=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Blade1.stl'))
blade4=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Blade4.stl'))
mixer1=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Mixer1.stl'))
mixer4=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Mixer4.stl'))
mixingDrum=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'mixingDrum.stl'))
shaft=stl.mesh.Mesh.from_file(os.path.join(destMeshPath, 'Shaft.stl'))
parFile=open(os.path.join(destPathBase, '0', "rpMain.pckl"),"rb"); rpMain=pickle.load(parFile); parFile.close()

computeTimestep=1.0e-5
statusTimestep=1e-3
dumpTimeStep=1e-3

angle=(360.0/((rpMain/computeTimestep)/(dumpTimeStep/computeTimestep)))

rolloutStepSize=1
offset=0

startTime=offset+(rp["nrPastVelocities"]+1)*rolloutStepSize

timestep=rp["nrPastVelocities"]
for timestep in range(1, predictTimesteps+rp["nrPastVelocities"]+2):
  myiter=timestep-1
  angle=(360.0/((rpMain/computeTimestep)/(dumpTimeStep/computeTimestep)))
  rotMatIter=np.array([[1.0, 0.0, 0.0], [0.0, np.cos(((angle*myiter)*math.pi)/180), -np.sin(((angle*myiter)*math.pi)/180)], [0.0, np.sin(((angle*myiter)*math.pi)/180), np.cos(((angle*myiter)*math.pi)/180)]])
  rotBlade1=np.matmul(blade1.vectors, rotMatIter.T)
  rotBlade4=np.matmul(blade4.vectors, rotMatIter.T)
  rotMixer1=np.matmul(mixer1.vectors, rotMatIter.T)
  rotMixer4=np.matmul(mixer4.vectors, rotMatIter.T)
  rotShaft=np.matmul(shaft.vectors, rotMatIter.T)
  rotMixingDrum=mixingDrum.vectors
  rotMesh=np.vstack([rotShaft, rotBlade1, rotBlade4, rotMixer1, rotMixer4, rotMixingDrum])
  xSceneData=rotMesh
  
  
  
  tsModulo=modulo//rp["stepSize"]
  
  if myiter%tsModulo==0:
    points=vtk.vtkPoints()
    vertices=vtk.vtkCellArray()
    for i in range(0, xSceneData.shape[0]):
      vertices.InsertNextCell(3)
      for vind in range(0, xSceneData.shape[1]):
        ins=points.InsertNextPoint(xSceneData[i][vind])
        vertices.InsertCellPoint(ins)
    
    polydata=vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(vertices)
    
    writer=vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(os.path.join(rolloutFilePrefix, "wall_"+str((myiter*rp["stepSize"])+1)+".vtp"))
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    writer.SetDataModeToAscii()
    writer.Write()