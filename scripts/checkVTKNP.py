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

#python3 /system/user/mayr/git/bgnn/scripts/checkVTKNP.py hopper runs1 model
#python3 /system/user/mayr/git/bgnn/scripts/checkVTKNP.py -problem hopper -experiment runs1 -saveModelName model -seq 30 31 35

import os
import sys
import vtk
import vtk.util
import vtk.util.numpy_support
import numpy as np
import pickle
import argparse



#problem=sys.argv[1]
#experiment=sys.argv[2]
#saveModelName=sys.argv[3]
#sys.argv.extend(["hopper", "runs1", "model"])
plotModulo=1

parser=argparse.ArgumentParser()
parser.add_argument("-problem", required=True, help="", type=str)
parser.add_argument("-experiment", required=True, help="", type=str)
parser.add_argument("-saveModelName", required=True, help="", type=str)
parser.add_argument("-seq", help="", nargs='+', type=int, default=[30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
args=parser.parse_args()

problem=args.problem
experiment=args.experiment
saveModelName=args.saveModelName







simDirBase="/system/user/mayr-data/BGNN/"

runDirBase="/system/user/mayr-data/BGNNRuns/"
modelDirBase=os.path.join(runDirBase, "models")
predDirBase=os.path.join(runDirBase, "predictions")
trajDirBase=os.path.join(runDirBase, "trajectories")



simDir=os.path.join(simDirBase, problem, experiment)

modelDir=os.path.join(modelDirBase, problem, experiment, saveModelName)
predDir=os.path.join(predDirBase, problem, experiment)
trajDir=os.path.join(trajDirBase, problem, experiment, saveModelName)



f=open(os.path.join(modelDir, "parInfo.pckl"), "rb")
rp=pickle.load(f)
f.close()
startTime=rp["nrPastVelocities"]+1



x1all=True
x2all=True

for mysequence in args.seq:
  sequenceGroundTruth=str(mysequence)
  sequencePrediction=str(mysequence)

  particleFilesGroundTruth=[]
  baseDirGroundTruth=os.path.join(simDir, sequenceGroundTruth, "post")
  timepointsGroundTruth=sorted([int(x.split("_")[2].split(".vtp")[0]) for x in os.listdir(baseDirGroundTruth) if 'main_particles_' in x and 'boundingBox' not in x])
  stepSizeGroundTruth=timepointsGroundTruth[1]-timepointsGroundTruth[0]

  particleFilesPred=[]
  baseDirPred=os.path.join(predDir, sequencePrediction+"_"+str(startTime), saveModelName)
  timepointsPred=sorted([int(x.split("_")[1].split(".vtp")[0]) for x in os.listdir(baseDirPred) if 'pred_' in x])
  stepSize=timepointsPred[1]-timepointsPred[0]

  #alignment of time steps
  plotTimePoints0=[timepointsPred[x] for x in range(0, len(timepointsPred), plotModulo)]
  plotGroundTruthTimePoints0=[(x-1)*stepSizeGroundTruth for x in plotTimePoints0]
  plotGroundTruthTimePoints=[x for x in plotGroundTruthTimePoints0 if x<=max(timepointsGroundTruth)]
  plotTimePoints=[plotTimePoints0[x] for x in range(len(plotTimePoints0)) if x<len(plotGroundTruthTimePoints)]

  for i in range(0, len(plotTimePoints0)):
    if i<len(plotTimePoints):
      particleFilesGroundTruth.append(os.path.join(baseDirGroundTruth, "main_particles_"+str(plotGroundTruthTimePoints[i])+".vtp"))
      particleFilesPred.append(os.path.join(baseDirPred, "pred_"+str(plotTimePoints[i])+".vtp"))
  
  timeDiff=np.array(plotTimePoints[1:])-np.array(plotTimePoints[:-1])
  assert(np.all(timeDiff==timeDiff[0]))
  cmpModulo=timeDiff[0]
  
  #each particle file in particleFilesPred, particleFilesGroundTruth is one (macro-)timestep

  gtList=[]
  predList=[]

  for i in range(0, len(particleFilesGroundTruth)):
    readFile=particleFilesGroundTruth[i]
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(readFile)
    reader.Update()
    simData=reader.GetOutputAsDataSet(0)

    gtData=vtk.util.numpy_support.vtk_to_numpy(simData.GetPoints().GetData())
    gtId=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("id"))
    gtData=gtData[np.argsort(gtId)]

    readFile=particleFilesPred[i]
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(readFile)
    reader.Update()
    simData=reader.GetOutputAsDataSet(0)

    predData=vtk.util.numpy_support.vtk_to_numpy(simData.GetPoints().GetData())
    predId=vtk.util.numpy_support.vtk_to_numpy(simData.GetAttributes(0).GetScalars("id"))
    
    if i==0:
      assert(np.all(gtData==predData))
    
    gtList.append(gtData)
    predList.append(predData)

  gtList=np.array(gtList)
  predList=np.array(predList)

  myf0=np.load(os.path.join(trajDir, "gtp_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  myf1=np.load(os.path.join(trajDir, "predp_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  
  
  print(mysequence)
  #print(baseDirPred)
  #print(os.path.join(trajDir, "predp_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  #print(predList.shape)
  #print(myf1.shape)
  #print(cmpModulo)
  print(np.all(myf0[np.arange(0,myf0.shape[0],cmpModulo)]==gtList))
  print(np.all(myf1[np.arange(0,myf1.shape[0],cmpModulo)]==predList))
  
  x1all=x1all and np.all(myf0[np.arange(0,myf0.shape[0],cmpModulo)]==gtList)
  x2all=x2all and np.all(myf1[np.arange(0,myf1.shape[0],cmpModulo)]==predList)

print("\n\nFinal:")
print(x1all)
print(x2all)
