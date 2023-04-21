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
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = "4"
import sys
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
import tensorflow as tf
from graph_nets import _base
from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
import sonnet as snt
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


parser=argparse.ArgumentParser()
parser.add_argument("-saveName", help="", type=str, default="model")
parser.add_argument("-addInfo", help="", type=str, default="")
parser.add_argument("-problem", help="", type=str, default="hopper")
parser.add_argument("-experiment", help="", type=str, default="runs2")
parser.add_argument("-plotSequence", help="", nargs='+', type=int, default=[0, 1, 2, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
parser.add_argument("-predictTimesteps", help="", type=int, default=2500)
parser.add_argument("-evalSequence", help="", nargs='+', type=int, default=[0, 1, 2, 30, 31, 32, 33, 34])
parser.add_argument("-evalTimesteps", help="", type=int, default=2500)
parser.add_argument("-modulo", help="", type=int, default=10)
parser.add_argument('--train', help='', type=eval, choices=[True, False], default='False')
parser.add_argument("--trainSaveInterval", help="", type=int, default=50000)
parser.add_argument('--execRollout', help='', type=eval, choices=[True, False], default='False')
parser.add_argument('--accScale', help='', type=eval, choices=[True, False], default='False')
parser.add_argument('--vtkOutput', help='', type=eval, choices=[True, False], default='True')
parser.add_argument('--npOutput', help='', type=eval, choices=[True, False], default='True')
parser.add_argument("--history-file", help="", type=str, default="")
parser.add_argument("--HistoryAccessor.enabled", help="", type=str, default="")
args=parser.parse_args()



tf.get_logger().setLevel(logging.ERROR)

#GIT
gitRoot=os.path.join(os.environ['HOME'], "git", "bgnn")
repo=git.Repo(gitRoot)
repoId=repo.head.object.hexsha
processGitDir=os.path.join(gitRoot, "bgnn", "tf")

#Script Tools
utilScript=os.path.join(processGitDir, "util.py")
wallInfoScript=os.path.join(processGitDir, "wallInfo_tf.py")
loadScript=os.path.join(processGitDir, "loadModel.py")
saveScript=os.path.join(processGitDir, "saveModel.py")
rolloutScript=os.path.join(processGitDir, "rolloutStatic.py")

exec(open(utilScript).read(), globals())

rp=collections.OrderedDict()
rp["version"]="processHopper2"
rp["problem"]="hopper"
rp["experiment"]="runs2"
rp["stepSize"]=1
rp["noiseConstraint"]="V1"
rp["noisyGraph"]=True
rp["noisyTarget"]=True
rp["ZNoise"]=0
rp["targetNormVel"]=(2,2,1,0)
rp["targetNormAcc"]=(2,2,1,0)
rp["wallOptImp"]=0.0
rp["optConstraintParticles"]=0
rp["optCrit"]=0
rp["constrainPredictionLength"]=0
rp["nrPastVelocities"]=5
rp["individualCutoff"]=False
rp["neighborCutoff"]=0.008
rp["velNoise"]=(np.array([1.0, 1.0, 1.0])*1e-06*0.1).tolist()
rp["accNoise"]=(np.array([1.0, 1.0, 1.0])*1e-08*0.1).tolist()
rp["correctNoise"]=1
rp["batchSize"]=1
rp["lrSchedule"]=0
rp["lrSpec"]=(0.0001, 1e-6, 0.1, int(3e6))
rp["implementation"]=4
rp["wallWeight"]=1
rp["useSelfLoops"]=False
rp["usePastVelocitiesLen"]=(True,(2,2,1,0))
rp["usePastVelocitiesVec"]=(True,(2,2,1,0))
rp["usePastVelocitiesLen2"]=(False,(2,2,1,0))
rp["useWallDistLen"]=(False,(2,2,1,0))
rp["useWallDistVec"]=(False,(2,2,1,0))
rp["useWallInvDistLen"]=(False,(2,2,1,0))
rp["useWallInvDistVec"]=(False,(2,2,1,0))
rp["useWallInvDist2Vec"]=(False,(2,2,1,0))
rp["useWallInvDist2Len"]=(False,(2,2,1,0))
rp["useWallDistLenClip"]=(False,(2,2,1,0))
rp["useWallInvDistLenClip"]=(False,(2,2,1,0))
rp["useWallInvDistLen2Clip"]=(False,(2,2,1,0))
rp["useWallDistLenClipInv"]=(False,(2,2,1,0))
rp["useTPDistLen"]=(False,(2,2,1,0))
rp["useTPDistVec"]=(False,(2,2,1,0))
rp["useTPInvDistLen"]=(False,(2,2,1,0))
rp["useTPInvDistVec"]=(False,(2,2,1,0))
rp["useTPInvDist2Len"]=(False,(2,2,1,0))
rp["useTPInvDist2Vec"]=(False,(2,2,1,0))
rp["useDistLen"]=(False,(2,2,1,0))
rp["useDistVec"]=(False,(2,2,1,0))
rp["useInvDistLen"]=(True,(2,2,1,0))
rp["useInvDistVec"]=(True,(2,2,1,0))
rp["useInvDist2Len"]=(False,(2,2,1,0))
rp["useInvDist2Vec"]=(False,(2,2,1,0))
rp["useDistLenVMod"]=(False,(2,2,1,0))
rp["useDistVecVMod"]=(False,(2,2,1,0))
rp["useInvDistLenVMod"]=(False,(2,2,1,0))
rp["useInvDistVecVMod"]=(False,(2,2,1,0))
rp["useInvDist2LenVMod"]=(False,(2,2,1,0))
rp["useInvDist2VecVMod"]=(False,(2,2,1,0))
rp["useProjectedUnitDistLenSenders"]=(False,(2,2,1,0))
rp["useProjectedUnitDistVecSenders"]=(False,(2,2,1,0))
rp["useProjectedUnitDistLen2Senders"]=(False,(2,2,1,0))
rp["useProjectedUnitDistLenReceivers"]=(False,(2,2,1,0))
rp["useProjectedUnitDistVecReceivers"]=(False,(2,2,1,0))
rp["useProjectedUnitDistLen2Receivers"]=(False,(2,2,1,0))
rp["useProjectedPartDistLenSum"]=(False,(2,2,1,0))
rp["useProjectedPartDistVecSum"]=(False,(2,2,1,0))
rp["useProjectedPartDistLen2Sum"]=(False,(2,2,1,0))
rp["useUnitDistVec"]=(False,(2,2,1,0))
rp["useNormalVec"]=(True,(2,2,1,0))
rp["multNormV"]=10.0
rp["useOneHotPE"]=(True,(2,2,1,0))
rp["multParticle"]=1.0
rp["multWall"]=100.0
rp["useAngle"]=(False,(2,2,1,0))
rp["multAngle"]=0.0
rp["gravBias"]=False
rp["model"]="modelHopper2.py"
rp["confFile"]="confHopper2.py"
rp["options"]="featureOptions1.py"

exec(open(os.path.join(processGitDir, rp["confFile"])).read(), globals())

rp["networkPar"]["nrSteps"]=3

assert(rp["problem"]==args.problem)
assert(rp["experiment"]==args.experiment)
assert(rp["version"]=="processHopper2")
#rp["problem"]=args.problem
#rp["experiment"]=args.experiment
#rp["version"]="processHopper2"
#rp["model"]="modelHopper2.py"
#rp["confFile"]="confHopper2.py"



featureOptionsScript=os.path.join(processGitDir, rp["options"])


destPathBase="/system/user/mayr-data/BGNN/"+rp["problem"]+"/"+args.experiment

parTab=pd.read_csv(destPathBase+"/parameters.csv")









xParticleDataList=[]
radiusParticleDataList=[]
typeParticleDataList=[]
xSceneDataList=[]
maxParticleNoiseDataList=[]
maxParticleCoordNoisePosDataList=[]
maxParticleCoordNoiseNegDataList=[]



overallVelVecSum=KahanSumExtendedNP()
overallVelVecSum2=KahanSumExtendedNP()
overallAccVecSum=KahanSumExtendedNP()
overallAccVecSum2=KahanSumExtendedNP()
overallVelLenSum=ScalarKahanSumExtended()
overallVelLenSum2=ScalarKahanSumExtended()
overallAccLenSum=ScalarKahanSumExtended()
overallAccLenSum2=ScalarKahanSumExtended()
velElems=0
accElems=0

loadIndices=np.concatenate([np.arange(0, 30)])
for i in loadIndices:
  print(i)
  destPath=destPathBase+"/"+str(i)
  
  if not os.path.exists(os.path.join(destPath, "xData.npy")):
    destPath=destPathBase+"/"+str(0)
  
  xParticleData=np.load(os.path.join(destPath, "xData.npy"), mmap_mode="r")
  xParticleDataList.append(xParticleData)
  
  radiusParticleData=np.load(os.path.join(destPath, "radiusData.npy"), mmap_mode="r")
  radiusParticleDataList.append(radiusParticleData)
  
  typeParticleData=np.load(os.path.join(destPath, "typeData.npy"), mmap_mode="r")
  typeParticleDataList.append(typeParticleData)
  
  xSceneData=np.load(os.path.join(destPath, "triangleCoordDataNew.npy"), mmap_mode="r")
  xSceneDataList.append(xSceneData)
  
  if rp["noiseConstraint"]=="V2":
    maxParticleNoiseData=np.load(os.path.join(destPath, "particleWallDistV2.npy"), mmap_mode="r")
    maxParticleNoiseDataList.append(maxParticleNoiseData)
    maxParticleCoordNoisePosData=np.load(os.path.join(destPath, "particleWallDiffPosV2.npy"), mmap_mode="r")
    maxParticleCoordNoisePosDataList.append(maxParticleCoordNoisePosData)
    maxParticleCoordNoiseNegData=np.load(os.path.join(destPath, "particleWallDiffNegV2.npy"), mmap_mode="r")
    maxParticleCoordNoiseNegDataList.append(maxParticleCoordNoiseNegData)
  else:
    maxParticleNoiseData=np.load(os.path.join(destPath, "particleWallDistV4.npy"), mmap_mode="r")
    maxParticleNoiseDataList.append(maxParticleNoiseData)
    maxParticleCoordNoisePosData=np.load(os.path.join(destPath, "particleWallDiffPosV4.npy"), mmap_mode="r")
    maxParticleCoordNoisePosDataList.append(maxParticleCoordNoisePosData)
    maxParticleCoordNoiseNegData=np.load(os.path.join(destPath, "particleWallDiffNegV4.npy"), mmap_mode="r")
    maxParticleCoordNoiseNegDataList.append(maxParticleCoordNoiseNegData)
  
  
  
  if i<30:
    velVecSum=np.load(os.path.join(destPath, "overallVelSum.npy"))
    velVecSum2=np.load(os.path.join(destPath, "overallVelSum2.npy"))
    accVecSum=np.load(os.path.join(destPath, "overallAccSum.npy"))
    accVecSum2=np.load(os.path.join(destPath, "overallAccSum2.npy"))
    overallVelVecSum.add(velVecSum)
    overallVelVecSum2.add(velVecSum2)
    overallAccVecSum.add(accVecSum)
    overallAccVecSum2.add(accVecSum2)
    velLenSum=np.load(os.path.join(destPath, "overallVelSum_scalar.npy"))
    velLenSum2=np.load(os.path.join(destPath, "overallVelSum2_scalar.npy"))
    accLenSum=np.load(os.path.join(destPath, "overallAccSum_scalar.npy"))
    accLenSum2=np.load(os.path.join(destPath, "overallAccSum2_scalar.npy"))
    overallVelLenSum.add(velLenSum)
    overallVelLenSum2.add(velLenSum2)
    overallAccLenSum.add(accLenSum)
    overallAccLenSum2.add(accLenSum2)
    
    velElems=velElems+np.load(os.path.join(destPath, "VelElems_scalar.npy"))
    accElems=accElems+np.load(os.path.join(destPath, "AccElems_scalar.npy"))

overallVelVecSum=overallVelVecSum.readout()
overallVelVecSum2=overallVelVecSum2.readout()
overallAccVecSum=overallAccVecSum.readout()
overallAccVecSum2=overallAccVecSum2.readout()
overallVelLenSum=overallVelLenSum.readout()
overallVelLenSum2=overallVelLenSum2.readout()
overallAccLenSum=overallAccLenSum.readout()
overallAccLenSum2=overallAccLenSum2.readout()

meanVelVec=np.reshape(overallVelVecSum/velElems,(1,-1))
stdVelVec=np.reshape(np.sqrt((overallVelVecSum2/velElems-(meanVelVec.flatten()**2))),(1,-1))
meanAccVec=np.reshape(overallAccVecSum/accElems,(1,-1))
stdAccVec=np.reshape(np.sqrt((overallAccVecSum2/accElems-(meanAccVec.flatten()**2))),(1,-1))

meanVelLen=np.reshape(overallVelLenSum/velElems,(1,-1))
stdVelLen=np.reshape(np.sqrt((overallVelLenSum2/velElems-(meanVelLen.flatten()**2))),(1,-1))
meanAccLen=np.reshape(overallAccLenSum/accElems,(1,-1))
stdAccLen=np.reshape(np.sqrt((overallAccLenSum2/accElems-(meanAccLen.flatten()**2))),(1,-1))

stdVelVecPerturbated=np.sqrt(stdVelVec**2+np.array([rp["velNoise"]])**2)
stdAccVecPerturbated=np.sqrt(stdAccVec**2+np.array([rp["accNoise"]])**2)

stdVelLenPerturbated=np.sqrt(stdVelLen**2+np.sum(np.array([rp["velNoise"]])**2,1,keepdims=True))
stdAccLenPerturbated=np.sqrt(stdAccLen**2+np.sum(np.array([rp["accNoise"]])**2,1,keepdims=True))


#meanVel=meanVel*0.0
#stdVel=np.mean(stdVel)
#meanAcc=meanAcc*0.0
#stdAcc=np.mean(stdAcc)



maxEpochs=1000
maxEpochsAbort=maxEpochs
maxMBAbort=len(xParticleDataList[0:30])*(3001-(rp["nrPastVelocities"]+1))
breakComputation=False

v1=[]
v2=[]
v3=[]
v4=[]
startEpoch=0
if True:
  trainList=list(itertools.product(list(range(len(xParticleDataList[0:30]))), list(range((rp["nrPastVelocities"]+1)*rp["stepSize"], 3001))))
  #np.random.shuffle(trainList)
  maxMB=len(trainList)
  startTrainNr=0
  
  for trainNr in range(startTrainNr, min(maxMB, maxMBAbort), rp["batchSize"]):
    print(str(trainNr)+"/"+str(maxMB))
    lockStat=False
    nrParticlesList=[]
    nextPositionVecList=[]
    currentPositionVecList=[]
    currentVelocityVecList=[]
    nextPositionVecPerturbatedList=[]
    currentPositionVecPerturbatedList=[]
    currentVelocityVecPerturbatedList=[]
    inputDictList=[]
    targetFeaturesNormDictList=[]
    targetFeaturesPerturbatedNormDictList=[]
    
    for batchInd in range(0, min(rp["batchSize"], len(trainList)-trainNr)):
      #break
      #break
      randPair=trainList[trainNr+batchInd]
      randSequence=randPair[0]
      timestep=randPair[1]
      
      xSceneData=xSceneDataList[randSequence]
      xParticleData=xParticleDataList[randSequence]
      maxParticleNoiseData=maxParticleNoiseDataList[randSequence]
      maxParticleCoordNoisePosData=maxParticleCoordNoisePosDataList[randSequence]
      maxParticleCoordNoiseNegData=maxParticleCoordNoiseNegDataList[randSequence]
      nrParticles=xParticleData.shape[1]
      nrConstraints=xSceneData.shape[1]
      
      with tf.device('/device:GPU:0'):
        currentConstPositionVec=tf.identity(np.float32(xSceneData[timestep-1*rp["stepSize"],:])) #that of timestep (current particle at timestep-1 should be related to (known) constraints at timestep)
        
        prevPositionVec=tf.identity(np.float32(xParticleData[timestep-2*rp["stepSize"],:]))
        currentPositionVec=tf.identity(np.float32(xParticleData[timestep-1*rp["stepSize"],:]))
        nextPositionVec=tf.identity(np.float32(xParticleData[timestep,:]))
        recentPositionVecs=tf.identity(np.float32(xParticleData[(timestep-(rp["nrPastVelocities"]+1)*rp["stepSize"]):(timestep):rp["stepSize"]]))
        
        recentMaxParticleNoise=tf.identity(np.float32(maxParticleNoiseData[(timestep-(rp["nrPastVelocities"]+1)*rp["stepSize"]):(timestep):rp["stepSize"]]))
        nextMaxParticleNoise=tf.identity(np.float32(maxParticleNoiseData[timestep:(timestep+1)]))
        recentMaxParticleNoisePos=tf.identity(np.float32(maxParticleCoordNoisePosData[(timestep-(rp["nrPastVelocities"]+1)*rp["stepSize"]):(timestep):rp["stepSize"]]))
        nextMaxParticleNoisePos=tf.identity(np.float32(maxParticleCoordNoisePosData[timestep:(timestep+1)]))
        recentMaxParticleNoiseNeg=tf.identity(np.float32(maxParticleCoordNoiseNegData[(timestep-(rp["nrPastVelocities"]+1)*rp["stepSize"]):(timestep):rp["stepSize"]]))
        nextMaxParticleNoiseNeg=tf.identity(np.float32(maxParticleCoordNoiseNegData[timestep:(timestep+1)]))
      
      velocityFeatureVecs=recentPositionVecs[1:]-recentPositionVecs[:-1]
      currentVelocityVec=recentPositionVecs[-1]-recentPositionVecs[-2]
      currentVelocityLen=tf.reshape(tf.sqrt(tf.reduce_sum(currentVelocityVec**2, 1)), (-1,1))
      currentAccelerationVec=(nextPositionVec-recentPositionVecs[-1])-velocityFeatureVecs[-1]
      #currentAccelerationVec=nextPositionVec-2.0*recentPositionVecs[-1]+recentPositionVecs[-2]
      currentAccelerationLen=tf.reshape(tf.sqrt(tf.reduce_sum(currentAccelerationVec**2, 1)), (-1,1))
      
      prevPositionVecPerturbated=prevPositionVec
      currentPositionVecPerturbated=currentPositionVec
      nextPositionVecPerturbated=nextPositionVec
      recentPositionVecsPerturbated=recentPositionVecs
      
      velocityFeatureVecsPerturbated=recentPositionVecsPerturbated[1:]-recentPositionVecsPerturbated[:-1]
      currentVelocityVecPerturbated=recentPositionVecsPerturbated[-1]-recentPositionVecsPerturbated[-2]
      currentVelocityLenPerturbated=tf.reshape(tf.sqrt(tf.reduce_sum(currentVelocityVecPerturbated**2, 1)), (-1,1))
      currentAccelerationVecPerturbated=(nextPositionVecPerturbated-recentPositionVecsPerturbated[-1])-velocityFeatureVecsPerturbated[-1]
      #currentAccelerationVecPerturbated=nextPositionVecPerturbated-2.0*recentPositionVecsPerturbated[-1]+recentPositionVecsPerturbated[-2]
      currentAccelerationLenPerturbated=tf.reshape(tf.sqrt(tf.reduce_sum(currentAccelerationVecPerturbated**2, 1)), (-1,1))
      
      if rp["individualCutoff"]:
        if rp["noisyGraph"]:
          velStrength=tf.sqrt(tf.reduce_sum(currentVelocityVecPerturbated**2,1))
        else:
          velStrength=tf.sqrt(tf.reduce_sum(currentVelocityVec**2,1))
        useCutoff=rp["neighborCutoff"]+velStrength
      else:
        useCutoff=rp["neighborCutoff"]
      useCutoff=rp["neighborCutoff"]
      
      points=currentPositionVecPerturbated
      tcoord=currentConstPositionVec
      useCutoff=rp["neighborCutoff"]
      exec(open(wallInfoScript).read(), globals())
      useCutoff=rp["neighborCutoff"]
      srInd_1=takeDist[:,0].numpy()
      #srInd_2=(takeDist[:,1]+nrParticles).numpy()
      srInd_2=np.arange(len(srInd_1))+nrParticles
      wallParticleSenders=np.concatenate([srInd_2])
      wallParticleReceivers=np.concatenate([srInd_1])
      #uwcount=tf.unique_with_counts(srInd_1)
      #particleWallWeight=1.0/uwcount[2].numpy()[uwcount[1].numpy()].astype(np.float32)
      if rp["wallWeight"]==0:
        particleWallWeight=tf.ones([len(srInd_1), 1])
      elif rp["wallWeight"]==1:
        particleWallWeight=1.0/tf.gather(tf.tensor_scatter_nd_add(tf.zeros([nrParticles]), tf.reshape(srInd_1, (-1,1)), tf.ones([len(srInd_1)])), srInd_1)
        particleWallWeight=tf.reshape(particleWallWeight, (-1,1))
      nrWallParticles=len(srInd_1)
      nrWallParticleDistances=len(wallParticleSenders)
      
      if rp["implementation"]==1:
        if rp["noisyGraph"]:
          myTree=scipy.spatial.cKDTree(currentPositionVecPerturbated.numpy())
          recvTree=myTree.query_ball_point(currentPositionVecPerturbated.numpy(), useCutoff)
        else:
          myTree=scipy.spatial.cKDTree(currentPositionVec)
          recvTree=myTree.query_ball_point(currentPositionVec.numpy(), useCutoff)
        sendTree=[np.repeat(i, len(recvTree[i])) for i in range(len(recvTree))]
        interParticleReceivers=np.concatenate(recvTree)
        interParticleSenders=np.concatenate(sendTree)
      elif rp["implementation"]==2:
        if rp["noisyGraph"]:
          myTree=sklearn.neighbors.KDTree(currentPositionVecPerturbated.numpy())
          recvTree=myTree.query_radius(currentPositionVecPerturbated.numpy(), r=useCutoff)
        else:
          myTree=sklearn.neighbors.KDTree(currentPositionVec.numpy())
          recvTree=myTree.query_radius(currentPositionVec.numpy(), r=useCutoff)
        sendTree=[np.repeat(i, len(recvTree[i])) for i in range(len(recvTree))]
        interParticleReceivers=np.concatenate(recvTree)
        interParticleSenders=np.concatenate(sendTree)
      elif rp["implementation"]==3:
        if rp["noisyGraph"]:
          myTree=scipy.spatial.cKDTree(tf.reshape(tf.concat([currentPositionVecPerturbated, currentPositionVecPerturbated+currentVelocityVecPerturbated],1), (-1,3)).numpy())
          recvTree=myTree.query_ball_point(currentPositionVecPerturbated.numpy(), useCutoff)
        else:
          myTree=scipy.spatial.cKDTree(tf.reshape(tf.concat([currentPositionVec, currentPositionVec+currentVelocityVec],1), (-1,3)).numpy())
          recvTree=myTree.query_ball_point(currentPositionVec.numpy(), useCutoff)
        recvTree=[[y//2 for y in x] for x in recvTree]
        recvTree=[[x[0]]+[x[i] for i in range(1,len(x)) if x[i]!=x[i-1]] for x in recvTree]
        sendTree=[np.repeat(i, len(recvTree[i])) for i in range(len(recvTree))]
        interParticleReceivers=np.concatenate(recvTree)
        interParticleSenders=np.concatenate(sendTree)
      elif rp["implementation"]==4:
        useCutoff=tf.identity(useCutoff)
        useCutoff=tf.reshape(useCutoff,-1)
        matTake=tf.where(tf.sqrt(tf.reduce_sum((tf.expand_dims(currentPositionVecPerturbated,0)-tf.expand_dims(currentPositionVecPerturbated,1))**2,2))<0.5*(tf.expand_dims(useCutoff,0)+tf.expand_dims(useCutoff,1)))
        interParticleReceivers=matTake[:,0]
        interParticleSenders=matTake[:,1]
        interParticleReceivers=interParticleReceivers.numpy()
        interParticleSenders=interParticleSenders.numpy()
      
      if not rp["useSelfLoops"]:
        commMask=interParticleReceivers!=interParticleSenders
        interParticleReceivers=interParticleReceivers[commMask]
        interParticleSenders=interParticleSenders[commMask]
      nrParticleDistances=len(interParticleSenders)
      
      
      
      senders=np.concatenate([interParticleSenders, wallParticleSenders])
      receivers=np.concatenate([interParticleReceivers, wallParticleReceivers])
      
      print("nrParticles: "+str(nrParticles))
      print("wallParticleSenders: "+str(len(wallParticleSenders)))
      print("nrParticleDistances: "+str(nrParticleDistances))
      print("len(senders): "+str(len(senders)))
      
      v1.append(nrParticles)
      v2.append(len(wallParticleSenders))
      v3.append(nrParticleDistances)
      v4.append(len(senders))



homeDir=os.getenv('HOME')
infoDir=os.path.join(os.environ['HOME'], "bgnnInfo")

import pickle
myf=open(os.path.join(infoDir, "hopper2Stat.pckl"), "wb")
pickle.dump(v1, myf)
pickle.dump(v2, myf)
pickle.dump(v3, myf)
pickle.dump(v4, myf)
myf.close()

