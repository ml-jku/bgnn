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
#sys.argv.extend(["-problem hopper -experiment runs1 -saveModelName model -seq 30"])
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
seq=args.seq


#problem="hopper"
#experiment="runs1"
#saveModelName="model"
#seq=[30, 31, 35]



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
x3all=True
x4all=True

for mysequence in seq:
  sequenceGroundTruth=str(mysequence)
  sequencePrediction=str(mysequence)

  particleFilesGroundTruth=[]
  wallFilesGroundTruth=[]
  baseDirGroundTruth=os.path.join(simDir, sequenceGroundTruth, "post")
  timepointsGroundTruth=sorted([int(x.split("_")[2].split(".vtp")[0]) for x in os.listdir(baseDirGroundTruth) if 'main_particles_' in x and 'boundingBox' not in x])
  stepSizeGroundTruth=timepointsGroundTruth[1]-timepointsGroundTruth[0]

  particleFilesPred=[]
  wallFilesPred=[]
  baseDirPred=os.path.join(predDir, sequencePrediction+"_"+str(startTime), saveModelName)
  timepointsPred=sorted([int(x.split("_")[1].split(".vtp")[0]) for x in os.listdir(baseDirPred) if 'pred_' in x])
  stepSize=timepointsPred[1]-timepointsPred[0]

  plotTimePoints0=[timepointsPred[x] for x in range(0, len(timepointsPred), plotModulo)]
  plotGroundTruthTimePoints0=[(x-1)*stepSizeGroundTruth for x in plotTimePoints0]
  plotGroundTruthTimePoints=[x for x in plotGroundTruthTimePoints0 if x<=max(timepointsGroundTruth)]
  plotTimePoints=[plotTimePoints0[x] for x in range(len(plotTimePoints0)) if x<len(plotGroundTruthTimePoints)]

  for i in range(0, len(plotTimePoints0)):
    if i<len(plotTimePoints):
      particleFilesGroundTruth.append(os.path.join(baseDirGroundTruth, "main_particles_"+str(plotGroundTruthTimePoints[i])+".vtp"))
      particleFilesPred.append(os.path.join(baseDirPred, "pred_"+str(plotTimePoints[i])+".vtp"))
      wallFilesGroundTruth.append(os.path.join(baseDirGroundTruth, "main_walls_"+str(plotGroundTruthTimePoints[i])+".vtp"))
      wallFilesPred.append(os.path.join(baseDirPred, "wall_"+str(plotTimePoints[i])+".vtp"))
  
  timeDiff=np.array(plotTimePoints[1:])-np.array(plotTimePoints[:-1])
  assert(np.all(timeDiff==timeDiff[0]))
  cmpModulo=timeDiff[0]
  
  #each particle file in particleFilesPred, particleFilesGroundTruth is one (macro-)timestep

  gtList=[]
  predList=[]
  gtWallList=[]
  predWallList=[]

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
    
    readFile=wallFilesGroundTruth[i]
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
    gtWallData=np.array(triangleCoords)
    
    readFile=wallFilesPred[i]
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
    predWallData=np.array(triangleCoords)
    
    
    
    if i==0:
      assert(np.all(gtData==predData))
    
    gtList.append(gtData)
    predList.append(predData)
    gtWallList.append(gtWallData)
    predWallList.append(predWallData)

  gtList=np.array(gtList)
  predList=np.array(predList)
  gtWallList=np.array(gtWallList)
  predWallList=np.array(predWallList)

  myf0=np.load(os.path.join(trajDir, "gtp_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  myf1=np.load(os.path.join(trajDir, "predp_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  myf2=np.load(os.path.join(trajDir, "gts_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  myf3=np.load(os.path.join(trajDir, "preds_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  
  print(mysequence)
  #print(baseDirPred)
  #print(os.path.join(trajDir, "predp_"+str(sequencePrediction)+"_"+str(startTime)+".npy"))
  #print(predList.shape)
  #print(myf1.shape)
  #print(cmpModulo)
  print(np.all(myf0[np.arange(0,myf0.shape[0],cmpModulo)]==gtList))
  print(np.all(myf1[np.arange(0,myf1.shape[0],cmpModulo)]==predList))
  print(np.all(myf2[np.arange(0,myf2.shape[0],cmpModulo)]==gtWallList))
  print(np.all(myf3[np.arange(0,myf3.shape[0],cmpModulo)]==predWallList))
  
  x1all=x1all and np.all(myf0[np.arange(0,myf0.shape[0],cmpModulo)]==gtList)
  x2all=x2all and np.all(myf1[np.arange(0,myf1.shape[0],cmpModulo)]==predList)
  x3all=x3all and np.all(myf2[np.arange(0,myf2.shape[0],cmpModulo)]==gtWallList)
  x4all=x4all and np.all(myf3[np.arange(0,myf3.shape[0],cmpModulo)]==predWallList)

print("\n\nFinal:")
print(x1all)
print(x2all)
print(x3all)
print(x4all)