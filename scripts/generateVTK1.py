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

#python3 /system/user/mayr/git/bgnn/scripts/generateVTK.py hopper runs1 model 10
#python3 /system/user/mayr/git/bgnn/scripts/generateVTK.py -problem hopper -experiment runs1 -saveModelName model -modulo 10 -seq 30 31 35

#/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs1/model
#/system/user/mayr-data/BGNNRuns/predictions/hopper/runs1/0_6


import numpy as np
import os
import sys
import vtk
import vtk.util
import vtk.util.numpy_support
import pickle
import argparse



#problem=sys.argv[1]
#experiment=sys.argv[2]
#saveModelName=sys.argv[3]
#modulo=int(sys.argv[4])

parser=argparse.ArgumentParser()
parser.add_argument("-problem", required=True, help="", type=str)
parser.add_argument("-experiment", required=True, help="", type=str)
parser.add_argument("-saveModelName", required=True, help="", type=str)
parser.add_argument("-modulo", help="", type=int, default=10)
parser.add_argument("-seq", help="", nargs='+', type=int, default=[30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
args=parser.parse_args()

problem=args.problem
experiment=args.experiment
saveModelName=args.saveModelName
modulo=args.modulo



saveFilePrefix="/system/user/mayr-data/BGNNRuns/models/"+problem+"/"+experiment+"/"+saveModelName+"/"
f=open(os.path.join(saveFilePrefix, "parInfo.pckl"), "rb")
rp=pickle.load(f)
f.close()

startTime=rp["nrPastVelocities"]+1

for randSequence in args.seq:
  rolloutFilePrefix=os.path.join("/system/user/mayr-data/BGNNRuns/predictions", rp["problem"], rp["experiment"], saveModelName, str(randSequence)+"_"+str(startTime))
  if not os.path.exists(os.path.join(rolloutFilePrefix)):
    os.makedirs(rolloutFilePrefix)
  
  
  
  xParticleData=np.load("/system/user/mayr-data/BGNNRuns/trajectories/"+rp["problem"]+"/"+rp["experiment"]+"/"+saveModelName+"/predp_"+str(randSequence)+"_"+str(startTime)+".npy")
  
  file=open(os.path.join("/system/user/mayr-data/BGNNRuns/trajectories/"+rp["problem"]+"/"+rp["experiment"]+"/"+saveModelName, "spgap_"+str(randSequence)+"_"+str(startTime)+".txt"), 'r')
  spgap=int(file.read())
  file.close()
  
  destPathBase="/system/user/mayr-data/BGNN/"+rp["problem"]+"/"+rp["experiment"]
  destPath=destPathBase+"/"+str(randSequence)
  
  radiusParticleData=np.load(os.path.join(destPath, "radiusData.npy"), mmap_mode="r")
  
  typeParticleData=np.load(os.path.join(destPath, "typeData.npy"), mmap_mode="r")
  
  
  
  typeData=typeParticleData[0,:]
  radiusData=radiusParticleData[0,:]
  
  tsModulo=modulo//rp["stepSize"]
  print(rolloutFilePrefix)
  for ind in range(len(xParticleData)):
    if (ind*spgap)%tsModulo==0:
      print(f"Writing Outfile: {ind}", end="\r")
      xDataPred=xParticleData[ind]
      idDataPred=np.arange(xDataPred.shape[0])+1
      
      points=vtk.vtkPoints()
      vertices=vtk.vtkCellArray()
      for i in range(0, xDataPred.shape[0]):
        ins=points.InsertNextPoint(xDataPred[i])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(ins)
      
      idOutVTK=vtk.vtkIntArray()
      idOutVTK.SetName("id")
      for i in range(0, idDataPred.shape[0]):
        idOutVTK.InsertNextValue(idDataPred[i])
      
      typeOutVTK=vtk.vtkIntArray()
      typeOutVTK.SetName("type")
      for i in range(0, typeData.shape[0]):
        typeOutVTK.InsertNextValue(typeData[i])
      
      radiusOutVTK=vtk.vtkFloatArray()
      radiusOutVTK.SetName("radius")
      for i in range(0, radiusData.shape[0]):
        radiusOutVTK.InsertNextValue(radiusData[i])
      
      polydata=vtk.vtkPolyData()
      polydata.SetPoints(points)
      polydata.SetVerts(vertices)
      polydata.GetPointData().AddArray(idOutVTK)
      polydata.GetPointData().AddArray(typeOutVTK)
      polydata.GetPointData().AddArray(radiusOutVTK)
      
      writer=vtk.vtkXMLPolyDataWriter()
      writer.SetFileName(os.path.join(rolloutFilePrefix, "pred_"+str((ind*spgap*rp["stepSize"])+1)+".vtp"))
      writer.SetInputData(polydata)
      writer.SetDataModeToBinary()
      writer.SetDataModeToAscii()
      writer.Write()
      
      
      
  print()