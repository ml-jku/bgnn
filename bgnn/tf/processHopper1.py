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
parser.add_argument("-experiment", help="", type=str, default="runs1")
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
rp["version"]="processHopper1"
rp["problem"]="hopper"
rp["experiment"]="runs1"
rp["stepSize"]=1
rp["noiseConstraint"]="V1"
rp["noisyGraph"]=True
rp["noisyTarget"]=False
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
rp["useDistLen"]=(True,(2,2,1,0))
rp["useDistVec"]=(True,(2,2,1,0))
rp["useInvDistLen"]=(False,(2,2,1,0))
rp["useInvDistVec"]=(False,(2,2,1,0))
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
rp["multNormV"]=100.0
rp["useOneHotPE"]=(True,(2,2,1,0))
rp["multParticle"]=1.0
rp["multWall"]=100.0
rp["useAngle"]=(False,(2,2,1,0))
rp["multAngle"]=0.0
rp["gravBias"]=False
rp["model"]="modelHopper1.py"
rp["confFile"]="confHopper1.py"
rp["options"]="featureOptions1.py"

restart=True if args.saveName!="" else False
if not restart:
  exec(open(os.path.join(processGitDir, rp["confFile"])).read(), globals())
  
  rp["networkPar"]["nrSteps"]=5
  
  saveDir="/system/user/mayr-data/BGNNRuns/models/"+args.problem+"/"+args.experiment+"/"
  hostname=os.uname()[1]
  hostname=socket.gethostname()
  mylock=open(os.path.join(saveDir, "mylock"), 'w')
  fcntl.flock(mylock, fcntl.LOCK_EX)
  contNr=max([int(x.split('_')[0].split('d')[1]) for x in [y for y in os.listdir(saveDir) if y.startswith('d')]+['d-1_']])+1
  saveName="d"+str(contNr)+"_"+str(int(time.time()*1000.0))+"_"+datetime.date.today().strftime("%d-%m-%Y")+"_hopper1"
  if args.addInfo!="":
    saveName=saveName+"_"+args.addInfo
  saveFilePrefix=saveDir+saveName+"/"
  if not os.path.exists(os.path.join(saveFilePrefix)):
    os.makedirs(saveFilePrefix)
  mylock.close()
else:
  saveDir="/system/user/mayr-data/BGNNRuns/models/"+args.problem+"/"+args.experiment+"/"
  saveName=args.saveName
  hostname=os.uname()[1]
  hostname=socket.gethostname()
  #contNr=int(saveName.split("_")[0][1:])
  saveFilePrefix=saveDir+saveName+"/"
  
  f=open(os.path.join(saveFilePrefix, "parInfo.pckl"), "rb")
  rpSave=pickle.load(f)
  f.close()
  rp.update(rpSave)

assert(rp["problem"]==args.problem)
assert(rp["experiment"]==args.experiment)
assert(rp["version"]=="processHopper1")
#rp["problem"]=args.problem
#rp["experiment"]=args.experiment
#rp["version"]="processHopper1"
#rp["model"]="modelHopper1.py"
#rp["confFile"]="confHopper1.py"



featureOptionsScript=os.path.join(processGitDir, rp["options"])



if args.train:
  systemOverfiew=open("/system/user/mayr-data/BGNNRuns/info/overview", "a")
  systemOverfiew.writelines(saveDir+" "+saveName+" "+hostname+" "+os.environ['CUDA_VISIBLE_DEVICES']+"\n")
  systemOverfiew.close()
  
  np.set_printoptions(suppress=False)
  infoFile=open("/system/user/mayr-data/BGNNRuns/info/"+saveName+"Extended", "w")
  #writeString=pprint.pformat(rp)
  writeString=json.dumps(rp, indent=2)
  print(writeString, file=infoFile)
  infoFile.close()
  
  #np.set_printoptions(suppress=True)



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

if args.train:
  loadIndices=np.concatenate([np.arange(0, 40)])
  #loadIndices=rp["loadIndices"]
else:
  loadIndices=np.arange(0, 40)



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



bstat=Statistics()
estat=Statistics()





modelScript=os.path.join(processGitDir, rp["model"])
exec(open(os.path.join(processGitDir, "blocks.py")).read(), globals())
exec(open(modelScript).read(), globals())
networkParMod=rp["networkPar"].copy()
if "output_size" in networkParMod:
  networkParMod["output_size"]=3
elif "node_output_size" in networkParMod:
  networkParMod["node_output_size"]=3
elif "outputPar" in networkParMod:
  networkParMod["outputPar"]["nodep"]["layerSz"][-1]=3
if rp["constrainPredictionLength"]==2 or rp["constrainPredictionLength"]==3:
  if "output_size" in networkParMod:
    networkParMod["output_size"]=4
  elif "node_output_size" in networkParMod:
    networkParMod["node_output_size"]=4
  elif "outputPar" in networkParMod:
    networkParMod["outputPar"]["nodep"]["layerSz"][-1]=4
networkParMod["nrTypes"]=2
myGN=EncodeProcessDecode(**networkParMod)



maxEpochs=1000
maxEpochsAbort=maxEpochs
maxMBAbort=len(xParticleDataList[0:30])*(2500-(rp["nrPastVelocities"]+1))
breakComputation=restart

optimizer=tf.keras.optimizers.Adam(learning_rate=rp["lrSpec"][0])
myoptSum=ScalarKahanSumExtended()
myoptN=0

if "epochNr" in dir():
  startEpoch=epochNr
  if "trainNr" in dir() and trainNr==min(len(trainList), maxMBAbort)-1:
    startEpoch=startEpoch+1
else:
  startEpoch=0
for epochNr in range(startEpoch, min(maxEpochs, maxEpochsAbort)):
  if "trainNr" in dir() and trainNr!=min(len(trainList), maxMBAbort)-1:
    maxMB=len(trainList)
    startTrainNr=trainNr+1
  else:
    trainList=list(itertools.product(list(range(len(xParticleDataList[0:30]))), list(range((rp["nrPastVelocities"]+1)*rp["stepSize"], 2500))))
    np.random.shuffle(trainList)
    maxMB=len(trainList)
    startTrainNr=0

  for trainNr in range(startTrainNr, min(maxMB, maxMBAbort), rp["batchSize"]):
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
      
      if rp["ZNoise"]<0.1:
        randomNoise=tf.concat([tf.zeros((1, nrParticles, 3)), tf.random.normal((rp["nrPastVelocities"], nrParticles, 3), mean=[0.,0.,0.], stddev=np.array(rp["velNoise"])/(rp["nrPastVelocities"] ** 0.5))], 0)
        randomNoise=tf.cumsum(randomNoise,0)
      if rp["ZNoise"]==1:
        randomNoise=tf.concat([tf.zeros((1, nrParticles, 3)), tf.random.normal((rp["nrPastVelocities"], nrParticles, 3), mean=[0.,0.,0.], stddev=np.array(rp["velNoise"])/(rp["nrPastVelocities"] ** 0.5))], 0)
        randomNoise=tf.concat([randomNoise[:,:,0:2], tf.abs(randomNoise[:,:,2:])*(tf.concat([tf.keras.backend.random_binomial((rp["nrPastVelocities"], nrParticles,1), 0.5), tf.keras.backend.random_binomial((1, nrParticles,1), 0.9)],0)*2-1)],2)
        randomNoise=tf.cumsum(randomNoise,0)
      elif rp["ZNoise"]>2 and rp["ZNoise"]<3:
        randomNoise=tf.concat([tf.zeros((1, nrParticles, 3)), tf.random.normal((rp["nrPastVelocities"], nrParticles, 3), mean=[0.,0.,0.], stddev=np.array(rp["velNoise"])/(rp["nrPastVelocities"] ** 0.5))], 0)
        randomNoise=tf.concat([randomNoise[:,:,0:2], tf.abs(randomNoise[:,:,2:])*(tf.concat([tf.keras.backend.random_binomial((rp["nrPastVelocities"], nrParticles,1), rp["ZNoise"]-2.0), tf.keras.backend.random_binomial((1, nrParticles,1), 0.9)],0)*2-1)],2)
        randomNoise=tf.cumsum(randomNoise,0)
      elif rp["ZNoise"]>3 and rp["ZNoise"]<=4:
        randomNoise=tf.concat([tf.zeros((1, nrParticles, 3)), tf.random.normal(shape=velocityFeatureVecs.shape, stddev=(rp["ZNoise"]-3.0)*tf.abs(velocityFeatureVecs))], 0)
      elif rp["ZNoise"]>4 and rp["ZNoise"]<=5:
        randomNoise=tf.concat([tf.zeros((1, nrParticles, 3)), tf.random.normal(shape=velocityFeatureVecs.shape, stddev=(rp["ZNoise"]-4.0)*tf.abs(velocityFeatureVecs))], 0)
        randomNoise=tf.cumsum(randomNoise,0)
      
      if rp["noisyTarget"]:
        randomNoise=tf.concat([randomNoise, randomNoise[-1:]],0)
      else:
        randomNoise=tf.concat([randomNoise, randomNoise[-1:]*0.0],0)
      
      maxNoise=tf.concat([recentMaxParticleNoise, nextMaxParticleNoise],0)
      maxNoisePos=tf.concat([recentMaxParticleNoisePos, nextMaxParticleNoisePos],0)
      maxNoiseNeg=tf.concat([recentMaxParticleNoiseNeg, nextMaxParticleNoiseNeg],0)
      noiseStrength=tf.sqrt(tf.reduce_sum(randomNoise**2,2))
      if rp["correctNoise"]==1:
        #all noise
        randomNoise=tf.where(np.logical_and(randomNoise>maxNoisePos, tf.expand_dims(noiseStrength>maxNoise, -1)), tf.random.uniform(maxNoisePos.shape, minval=0, maxval=maxNoisePos), randomNoise)
        randomNoise=tf.where(np.logical_and(randomNoise<maxNoiseNeg, tf.expand_dims(noiseStrength>maxNoise, -1)), tf.random.uniform(maxNoiseNeg.shape, minval=maxNoiseNeg, maxval=0), randomNoise)
      elif rp["correctNoise"]==2:
        #only target noise correction, no features noise correction
        randomNoiseCp=tf.where(np.logical_and(randomNoise>maxNoisePos, tf.expand_dims(noiseStrength>maxNoise, -1)), tf.random.uniform(maxNoisePos.shape, minval=0, maxval=maxNoisePos), randomNoise)
        randomNoiseCp=tf.where(np.logical_and(randomNoiseCp<maxNoiseNeg, tf.expand_dims(noiseStrength>maxNoise, -1)), tf.random.uniform(maxNoiseNeg.shape, minval=maxNoiseNeg, maxval=0), randomNoiseCp)
        randomNoise=tf.concat([randomNoise[0:6], randomNoiseCp[6:]],0)
      elif rp["correctNoise"]==3:
        #only target noise + current noise correction, no other features noise correction
        randomNoiseCp=tf.where(np.logical_and(randomNoise>maxNoisePos, tf.expand_dims(noiseStrength>maxNoise, -1)), tf.random.uniform(maxNoisePos.shape, minval=0, maxval=maxNoisePos), randomNoise)
        randomNoiseCp=tf.where(np.logical_and(randomNoiseCp<maxNoiseNeg, tf.expand_dims(noiseStrength>maxNoise, -1)), tf.random.uniform(maxNoiseNeg.shape, minval=maxNoiseNeg, maxval=0), randomNoiseCp)
        randomNoise=tf.concat([randomNoise[0:5], randomNoiseCp[5:]],0)
      
      prevPositionVecPerturbated=prevPositionVec+randomNoise[-3]
      currentPositionVecPerturbated=currentPositionVec+randomNoise[-2]
      nextPositionVecPerturbated=nextPositionVec+randomNoise[-1]
      recentPositionVecsPerturbated=recentPositionVecs+randomNoise[0:-1]
      
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
      
      points=currentPositionVecPerturbated
      tcoord=currentConstPositionVec
      exec(open(wallInfoScript).read(), globals())
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
      
      
      
      partDistVecPerturbated=-(tf.gather(currentPositionVecPerturbated, interParticleSenders)-tf.gather(currentPositionVecPerturbated, interParticleReceivers))
      partDistLenPerturbated=tf.reshape(tf.sqrt(tf.reduce_sum(partDistVecPerturbated**2, 1)), (-1,1))
      partUnitDistVecPerturbated=partDistVecPerturbated/partDistLenPerturbated
      partUnitDistVecPerturbated=tf.where(tf.math.is_finite(partUnitDistVecPerturbated), partUnitDistVecPerturbated, 0)
      
      wallDistVec=-tf.gather_nd(diffVec, takeDist)
      wallDistLen=tf.sqrt(tf.reshape(tf.reduce_sum(wallDistVec**2,1),(-1,1)))
      wallUnitDistVec=wallDistVec/wallDistLen
      wallUnitDistVec=tf.where(tf.math.is_finite(wallUnitDistVec), wallUnitDistVec, 0)
      
      projectedPartDistVecSendersPerturbated=tf.reshape(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, interParticleSenders)*partUnitDistVecPerturbated,1),(-1,1))*partUnitDistVecPerturbated
      projectedPartDistLenSendersPerturbated=tf.sqrt(tf.reduce_sum(projectedPartDistVecSendersPerturbated**2,1, keepdims=True))
      projectedPartDistVecReceiversPerturbated=tf.reshape(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, interParticleReceivers)*partUnitDistVecPerturbated,1),(-1,1))*partUnitDistVecPerturbated
      projectedPartDistLenReceiversPerturbated=tf.sqrt(tf.reduce_sum(projectedPartDistVecReceiversPerturbated**2,1, keepdims=True))
      projectedPartDistVecSumPerturbated=projectedPartDistVecSendersPerturbated-projectedPartDistVecReceiversPerturbated
      projectedPartDistLenSumPerturbated=tf.sqrt(tf.reduce_sum(projectedPartDistVecSumPerturbated**2,1, keepdims=True))
      
      projectedWallDistVecPerturbated=tf.reshape(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, takeDist[:,0])*wallUnitDistVec,1),(-1,1))*wallUnitDistVec
      projectedWallDistLenPerturbated=tf.sqrt(tf.reduce_sum(projectedWallDistVecPerturbated**2,1, keepdims=True))
      
      partDistVecVMod=(partDistVecPerturbated-projectedPartDistVecSumPerturbated)
      partDistLenVMod=tf.reshape(tf.sqrt(tf.reduce_sum(partDistVecVMod**2, 1)), (-1,1))
      
      wallDistVecVMod=(wallDistVec-projectedWallDistVecPerturbated)
      wallDistLenVMod=tf.sqrt(tf.reshape(tf.reduce_sum(wallDistVec**2,1),(-1,1)))
      
      angle=tf.abs(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, takeDist[:,0])*tf.gather(nvec, takeDist[:,1]),1)/(tf.sqrt(tf.reduce_sum((tf.gather(currentVelocityVecPerturbated, takeDist[:,0]))**2,1))*tf.sqrt(tf.reduce_sum((tf.gather(nvec, takeDist[:,1]))**2,1))))
      angle=tf.where(tf.math.is_nan(angle), 0, angle)
      angle=angle*rp["multAngle"]
      
      ucurrentConstPositionVec=np.unique(np.reshape(currentConstPositionVec,(-1,3)),axis=0)
      devDistVec=tf.reshape(ucurrentConstPositionVec,(-1,1,3))-tf.reshape(ucurrentConstPositionVec,(1,-1,3))
      devDistLen=tf.sqrt(tf.reduce_sum(devDistVec**2, 2, keepdims=True))
      devDistMax=0.5*tf.reduce_max(devDistLen)
      #devDistVec=tf.reshape(ucurrentConstPositionVec,(1,-1,3))-tf.reshape(tf.reduce_mean(currentConstPositionVec, 1),(-1,1,3))
      #devDistLen=tf.sqrt(tf.reduce_sum(devDistVec**2, 2, keepdims=True))
      #partDevDistVec=tf.reshape(ucurrentConstPositionVec,(1,-1,3))-tf.reshape(points,(-1,1,3))
      #partDevDistLen=tf.sqrt(tf.reduce_sum(partDevDistVec**2, 2, keepdims=True))
      #devDistLen=tf.gather(devDistLen, takeDist[:,1])
      #devDistVec=tf.gather(devDistVec, takeDist[:,1])
      partDevDistVec=tf.reshape(points,(-1,1,3))-tf.reshape(ucurrentConstPositionVec,(1,-1,3))
      partDevDistLen=tf.sqrt(tf.reduce_sum(partDevDistVec**2, 2, keepdims=True))
      wallPoints=tf.gather_nd(minPoint, takeDist)
      devDistVec=tf.reshape(wallPoints,(-1,1,3))-tf.reshape(ucurrentConstPositionVec,(1,-1,3))
      devDistLen=tf.sqrt(tf.reduce_sum(devDistVec**2, 2, keepdims=True))
      
      
      
      if epochNr==0:
        bstat.track("currentVelocityVec", currentVelocityVec)
        bstat.track("currentVelocityLen", currentVelocityLen)
        bstat.track("currentAccelerationVec", currentAccelerationVec)
        bstat.track("currentAccelerationLen", currentAccelerationLen)
        bstat.track("currentVelocityVecPerturbated", currentVelocityVecPerturbated)
        bstat.track("currentVelocityLenPerturbated", currentVelocityLenPerturbated)
        bstat.track("currentAccelerationVecPerturbated", currentAccelerationVecPerturbated)
        bstat.track("currentAccelerationLenPerturbated", currentAccelerationLenPerturbated)
        bstat.trackZeros("wallVelocityVec", tuple(currentVelocityVecPerturbated.shape[1:]), nrWallParticleDistances)
        bstat.trackZeros("wallVelocityLen", tuple(currentVelocityLenPerturbated.shape[1:]), nrWallParticleDistances)
        bstat.trackZeros("wallAccelerationVec", tuple(currentAccelerationVecPerturbated.shape[1:]), nrWallParticleDistances)
        bstat.trackZeros("wallAccelerationLen", tuple(currentAccelerationLenPerturbated.shape[1:]), nrWallParticleDistances)
        
        bstat.track("partDistVecPerturbated", partDistVecPerturbated)
        bstat.track("partDistLenPerturbated", partDistLenPerturbated)
        bstat.track("partUnitDistVecPerturbated", partUnitDistVecPerturbated)
        bstat.track("wallDistVec", wallDistVec)
        bstat.track("wallDistLen", wallDistLen)
        bstat.track("wallUnitDistVec", wallUnitDistVec)
        
        bstat.track("projectedPartDistVecSendersPerturbated", projectedPartDistVecSendersPerturbated)
        bstat.track("projectedPartDistLenSendersPerturbated", projectedPartDistLenSendersPerturbated)
        bstat.track("projectedPartDistVecReceiversPerturbated", projectedPartDistVecReceiversPerturbated)
        bstat.track("projectedPartDistLenReceiversPerturbated", projectedPartDistLenReceiversPerturbated)
        bstat.track("projectedPartDistVecSumPerturbated", projectedPartDistVecSumPerturbated)
        bstat.track("projectedPartDistLenSumPerturbated", projectedPartDistLenSumPerturbated)
        bstat.track("projectedWallDistVecPerturbated", projectedWallDistVecPerturbated)
        bstat.track("projectedWallDistLenPerturbated", projectedWallDistLenPerturbated)
        
        bstat.track("partDistVecVMod", partDistVecVMod)
        bstat.track("partDistLenVMod", partDistLenVMod)
        bstat.track("wallDistVecVMod", wallDistVecVMod)
        bstat.track("wallDistLenVMod", wallDistLenVMod)
        
        bstat.track("angle", angle)
        bstat.endTrack()
      
      
      
      if rp["noisyTarget"]:
        meanVelP, stdVelP, meanVelW, stdVelW=normParV(rp["targetNormVel"], bstat, "currentVelocityVecPerturbated", "wallVelocityVec", "currentVelocityLenPerturbated", "wallVelocityLen", meanVelVec, stdVelLenPerturbated, stdVelVecPerturbated)
        meanAccP, stdAccP, meanAccW, stdAccW=normParV(rp["targetNormAcc"], bstat, "currentAccelerationVecPerturbated", "wallAccelerationVec", "currentAccelerationLenPerturbated", "wallAccelerationLen", meanAccVec, stdAccLenPerturbated, stdAccVecPerturbated)
      else:
        meanVelP, stdVelP, meanVelW, stdVelW=normParV(rp["targetNormVel"], bstat, "currentVelocityVec", "wallVelocityVec", "currentVelocityLen", "wallVelocityLen", meanVelVec, stdVelLen, stdVelVec)
        meanAccP, stdAccP, meanAccW, stdAccW=normParV(rp["targetNormAcc"], bstat, "currentAccelerationVec", "wallAccelerationVec", "currentAccelerationLen", "wallAccelerationLen", meanAccVec, stdAccLen, stdAccVec)
      
      
      
      #velocityFeatureVecsNorm=(velocityFeatureVecs-meanVel)/stdVel
      #velocityFeatureVecsPerturbatedNorm=(velocityFeatureVecsPerturbated-meanVel)/stdVel
      currentAccelerationVecNormP=(currentAccelerationVec-meanAccP)/stdAccP
      currentAccelerationVecPerturbatedNormP=(currentAccelerationVecPerturbated-meanAccP)/stdAccP
      currentAccelerationVecNormW=(tf.zeros((nrWallParticles, 3))-meanAccW)/stdAccW
      
      if rp["gravBias"]:
        currentAccelerationVecNormP=currentAccelerationVecNormP-tf.identity(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32))/stdAccP
        currentAccelerationVecPerturbatedNormP=currentAccelerationVecPerturbatedNormP-tf.identity(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32))/stdAccP
        currentAccelerationVecNormW=currentAccelerationVecNormW-tf.identity(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32))/stdAccW
      
      
      exec(open(featureOptionsScript).read(), globals())
      
      
      if len(particleNodeDataList)<2:
        particleNodeDataList.append(tf.ones((nrParticles,1)))
        particleNodeDataList.append(-tf.ones((nrParticles,1)))
        wallNodeDataList.append( tf.ones((nrWallParticles,1)))
        wallNodeDataList.append(-tf.ones((nrWallParticles,1)))
      
      #for particle embedding
      #particleNodeDataList.append(tf.constant(typeParticleDataList[randSequence][timestep-1*rp["stepSize"]]-1, dtype=tf.float32, shape=(nrParticles,1)))
      particleNodeDataList.append(tf.constant(1.0, dtype=tf.float32, shape=(nrParticles,1)))
      wallNodeDataList.append(tf.constant(0.0, dtype=tf.float32, shape=(nrWallParticles,1)))
      
      if len(particleEdgeDataList)<2:
        particleEdgeDataList.append(tf.ones((nrParticleDistances,1)))
        particleEdgeDataList.append(-tf.ones((nrParticleDistances,1)))
        wallEdgeDataList.append( tf.ones((nrWallParticleDistances,1)))
        wallEdgeDataList.append(-tf.ones((nrWallParticleDistances,1)))
      
      particleNodeData=tf.concat(particleNodeDataList, 1)
      wallNodeData=tf.concat(wallNodeDataList, 1)
      particleEdgeData=tf.concat(particleEdgeDataList, 1)
      wallEdgeData=tf.concat(wallEdgeDataList, 1)
      
      nodeFeatures=tf.concat([particleNodeData, wallNodeData], 0)
      edgeFeatures=tf.concat([particleEdgeData, wallEdgeData], 0)
      senders=np.concatenate([interParticleSenders, wallParticleSenders])
      receivers=np.concatenate([interParticleReceivers, wallParticleReceivers])
      
      
      
      inputDict={
        "nodes": nodeFeatures,
        "edges": edgeFeatures,
        "senders": senders,
        "receivers": receivers
      }
      
      
      
      
      if rp["wallOptImp"]>0.0:
        importance=rp["wallOptImp"]*tf.reshape((tf.scatter_nd(takeDist[:,0:1], bstat.getStd("wallDistLen")[0]/(wallDistLen[:,0]+0.01*bstat.getStd("wallDistLen")[0]), shape=[nrParticles])), (-1,1))
        targetFeaturesNorm=tf.concat([tf.concat([currentAccelerationVecNormP, currentAccelerationVecNormW], 0), 
                                      tf.concat([tf.ones((nrParticles,1))+importance, tf.zeros((nrWallParticles,1))], 0)
                                      ], 1)
        targetFeaturesPerturbatedNorm=tf.concat([tf.concat([currentAccelerationVecPerturbatedNormP, currentAccelerationVecNormW], 0), 
                                                 tf.concat([tf.ones((nrParticles,1))+importance, tf.zeros((nrWallParticles,1))], 0)
                                                 ], 1)
      else:
        targetFeaturesNorm=tf.concat([tf.concat([currentAccelerationVecNormP, currentAccelerationVecNormW], 0), 
                                      tf.concat([tf.ones((nrParticles,1)), tf.zeros((nrWallParticles,1))], 0)
                                      ], 1)
        targetFeaturesPerturbatedNorm=tf.concat([tf.concat([currentAccelerationVecPerturbatedNormP, currentAccelerationVecNormW], 0), 
                                                 tf.concat([tf.ones((nrParticles,1)), tf.zeros((nrWallParticles,1))], 0)
                                                 ], 1)
      
      targetFeaturesNormDict={
        "nodes": targetFeaturesNorm
      }
      targetFeaturesPerturbatedNormDict={
        "nodes": targetFeaturesPerturbatedNorm
      }
      
      nrParticlesList.append(nrParticles)
      nextPositionVecList.append(nextPositionVec)
      currentPositionVecList.append(currentPositionVec)
      currentVelocityVecList.append(currentVelocityVec)
      nextPositionVecPerturbatedList.append(nextPositionVecPerturbated)
      currentPositionVecPerturbatedList.append(currentPositionVecPerturbated)
      currentVelocityVecPerturbatedList.append(currentVelocityVecPerturbated)
      inputDictList.append(inputDict)
      targetFeaturesNormDictList.append(targetFeaturesNormDict)
      targetFeaturesPerturbatedNormDictList.append(targetFeaturesPerturbatedNormDict)
    
    graphInputBatch=utils_tf.data_dicts_to_graphs_tuple(inputDictList)
    if rp["noisyTarget"]:
      graphOutputBatch=utils_tf.data_dicts_to_graphs_tuple(targetFeaturesPerturbatedNormDictList)
    else:
      graphOutputBatch=utils_tf.data_dicts_to_graphs_tuple(targetFeaturesNormDictList)
    
    with tf.GradientTape() as tape:
      myresBatch=myGN(graphInputBatch)
      if rp["constrainPredictionLength"]==1:
        myresBatch=myresBatch.replace(nodes=tf.transpose(tf.stack([(myresBatch.nodes[:,0])*rp["neighborCutoff"]*tf.sin(tf.nn.sigmoid(myresBatch.nodes[:,1])*np.pi*1.5-np.pi*0.25)*tf.cos(tf.nn.sigmoid(myresBatch.nodes[:,2])*np.pi*2.0*1.5-np.pi*2.0*0.25),
        (myresBatch.nodes[:,0])*rp["neighborCutoff"]*tf.sin(tf.nn.sigmoid(myresBatch.nodes[:,1])*np.pi*1.5-np.pi*0.25)*tf.sin(tf.nn.sigmoid(myresBatch.nodes[:,2])*np.pi*2.0*1.5-np.pi*2.0*0.25),
        (myresBatch.nodes[:,0])*rp["neighborCutoff"]*tf.cos(tf.nn.sigmoid(myresBatch.nodes[:,1])*np.pi*1.5-np.pi*0.25)])))
      elif rp["constrainPredictionLength"]==2:
        normVector=tf.sqrt(tf.reshape(tf.reduce_sum(myresBatch.nodes[:,0:3]**2,1),(-1,1)))
        dirVector=myresBatch.nodes[:,0:3]/normVector
        myresBatch=myresBatch.replace(nodes=dirVector*(myresBatch.nodes[:,3:4])*rp["neighborCutoff"])
      elif rp["constrainPredictionLength"]==3:
        with tape.stop_recording():
          normVector=tf.sqrt(tf.reshape(tf.reduce_sum(myresBatch.nodes[:,0:3]**2,1),(-1,1)))
        dirVector=myresBatch.nodes[:,0:3]/normVector
        myresBatch=myresBatch.replace(nodes=dirVector*(myresBatch.nodes[:,3:4])*rp["neighborCutoff"])
      
      if rp["optConstraintParticles"]==0:
        if rp["optCrit"]==0:
          optCrit=tf.reduce_sum(tf.math.reduce_mean(((myresBatch.nodes-graphOutputBatch.nodes[:,0:3])**2)*graphOutputBatch.nodes[:,3:4], 1))/tf.reduce_sum(graphOutputBatch.nodes[:,3:4])
        elif rp["optCrit"]==1:
          strength=tf.sqrt(tf.reduce_sum(graphOutputBatch.nodes[:,0:3]**2,1, keepdims=True))
          optCrit0=((myresBatch.nodes-graphOutputBatch.nodes[:,0:3])**2)*graphOutputBatch.nodes[:,3:4]
          optCrit1=((((myresBatch.nodes-graphOutputBatch.nodes[:,0:3])**2)/strength)*graphOutputBatch.nodes[:,3:4])/tf.reduce_sum(graphOutputBatch.nodes[:,3:4])
          optCrit=tf.reduce_sum(tf.math.reduce_mean(tf.where(strength==0.0, optCrit0, optCrit1)))/tf.reduce_sum(graphOutputBatch.nodes[:,3:4])
      elif rp["optConstraintParticles"]==1:
        optCrit=tf.reduce_mean(tf.math.reduce_mean(((myresBatch.nodes-graphOutputBatch.nodes[:,0:3])**2), 1))
      elif rp["optConstraintParticles"]==2:
        optCrit=tf.reduce_sum(tf.math.reduce_mean(((myresBatch.nodes-graphOutputBatch.nodes[:,0:3])**2)*graphOutputBatch.nodes[:,3:4], 1))/tf.reduce_sum(graphOutputBatch.nodes[:,3:4])+tf.reduce_sum(tf.math.reduce_mean(((myresBatch.nodes-graphOutputBatch.nodes[:,0:3])**2)*(1-graphOutputBatch.nodes[:,3:4]), 1))/tf.reduce_sum(1-graphOutputBatch.nodes[:,3:4])
      myopt=optCrit.numpy()
    gradients=tape.gradient(optCrit, myGN.trainable_variables)
    
    if rp["lrSchedule"]==1:
      stepLR=min(maxMB, maxMBAbort)*epochNr+trainNr
      if (len(rp["lrSpec"])>4.5 and stepLR>rp["lrSpec"][4]) or (epochNr>0.5):
        baseLR=rp["lrSpec"][0]
        minLR=rp["lrSpec"][1]
        decayLR=rp["lrSpec"][2]
        stepDecay=rp["lrSpec"][3]
        decayedLR=((baseLR-minLR)*(decayLR**(stepLR / stepDecay)))+minLR
        optimizer.lr.assign(decayedLR)
    
    #optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    optimizer.apply_gradients(zip(gradients, myGN.trainable_variables))
    if "norm" in dir(myGN):
      myGN.norm()
    
    myresBatch=myGN(graphInputBatch)
    for batchInd in range(0, rp["batchSize"]):
      nrParticles=nrParticlesList[batchInd]
      nextPositionVec=nextPositionVecList[batchInd]
      currentPositionVec=currentPositionVecList[batchInd]
      currentVelocityVec=currentVelocityVecList[batchInd]
      nextPositionVecPerturbated=nextPositionVecPerturbatedList[batchInd]
      currentPositionVecPerturbated=currentPositionVecPerturbatedList[batchInd]
      currentVelocityVecPerturbated=currentVelocityVecPerturbatedList[batchInd]
      myres=utils_tf.get_graph(myresBatch, batchInd)
      if rp["constrainPredictionLength"]==1:
        myres=myres.replace(nodes=tf.transpose(tf.stack([(myres.nodes[:,0])*rp["neighborCutoff"]*tf.sin(tf.nn.sigmoid(myres.nodes[:,1])*np.pi*1.5-np.pi*0.25)*tf.cos(tf.nn.sigmoid(myres.nodes[:,2])*np.pi*2.0*1.5-np.pi*2.0*0.25),
        (myres.nodes[:,0])*rp["neighborCutoff"]*tf.sin(tf.nn.sigmoid(myres.nodes[:,1])*np.pi*1.5-np.pi*0.25)*tf.sin(tf.nn.sigmoid(myres.nodes[:,2])*np.pi*2.0*1.5-np.pi*2.0*0.25),
        (myres.nodes[:,0])*rp["neighborCutoff"]*tf.cos(tf.nn.sigmoid(myres.nodes[:,1])*np.pi*1.5-np.pi*0.25)])))
      elif rp["constrainPredictionLength"]==2:
        normVector=tf.sqrt(tf.reshape(tf.reduce_sum(myres.nodes[:,0:3]**2,1),(-1,1)))
        dirVector=myres.nodes[:,0:3]/normVector
        myres=myres.replace(nodes=dirVector*(myres.nodes[:,3:4])*rp["neighborCutoff"])
      elif rp["constrainPredictionLength"]==3:
        normVector=tf.sqrt(tf.reshape(tf.reduce_sum(myres.nodes[:,0:3]**2,1),(-1,1)))
        dirVector=myres.nodes[:,0:3]/normVector
        myres=myres.replace(nodes=dirVector*(myres.nodes[:,3:4])*rp["neighborCutoff"])
      if False: #rp["gravBias"]:
        predAcceleration=(myres.nodes[:nrParticles]*stdAccP)+meanAccP+tf.identity(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32))
      else:
        predAcceleration=(myres.nodes[:nrParticles]*stdAccP)+meanAccP
      #s1: 0.5*g*t^2
      #s2: 0.5*g*(2*t)^2
      #s3: 0.5*g*(3*t)^2
      #v1: s2-s1=0.5*g*3*t^2
      #v2: s3-s2=0.5*g*5*t^2
      #v2-v1=0.5*g*2*t^2=g*t^2
      
      #s1: 0.0
      #s2: 0.5*g*(t)^2
      #s3: 0.5*g*(2*t)^2
      #v1: s2-s1=0.5*g*t^2
      #v2: s3-s2=0.5*g*3*t^2
      #v2-v1=0.5*g*2*t^2=g*t^2
      
      #s1: 0.5*g*(2*t)^2
      #s2: 0.5*g*(3*t)^2
      #s3: 0.5*g*(4*t)^2
      #v1: s2-s1=0.5*g*5*t^2
      #v2: s3-s2=0.5*g*7*t^2
      #v2-v1=0.5*g*2*t^2=g*t^2
      
      
      
      
      optCritCmp00=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVecPerturbated+1.0*(currentVelocityVecPerturbated+1.0*(predAcceleration)))-nextPositionVecPerturbated)**2, 1))
      optCritCmp10=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVecPerturbated+1.0*(currentVelocityVecPerturbated))-nextPositionVecPerturbated)**2, 1))
      optCritCmp20=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVecPerturbated)-nextPositionVecPerturbated)**2, 1))
      
      optCritCmp01=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVec+1.0*(currentVelocityVec+1.0*(predAcceleration)))-nextPositionVec)**2, 1))
      optCritCmp11=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVec+1.0*(currentVelocityVec))-nextPositionVec)**2, 1))
      optCritCmp21=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVec)-nextPositionVec)**2, 1))
      
      optCritCmp02=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVecPerturbated+1.0*(currentVelocityVecPerturbated+1.0*(predAcceleration)))-nextPositionVec)**2, 1))
      optCritCmp12=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVecPerturbated+1.0*(currentVelocityVecPerturbated))-nextPositionVec)**2, 1))
      optCritCmp22=tf.reduce_mean(tf.math.reduce_mean(((currentPositionVecPerturbated)-nextPositionVec)**2, 1))
      
      printStr=f'{myopt:>15.10f} |'\
               f'{optCritCmp00.numpy():10.3E},'\
               f'{optCritCmp10.numpy():10.3E},'\
               f'{optCritCmp20.numpy():10.3E} |'\
               f'{optCritCmp01.numpy():10.3E},'\
               f'{optCritCmp11.numpy():10.3E},'\
               f'{optCritCmp21.numpy():10.3E} |'\
               f'{optCritCmp02.numpy():10.3E},'\
               f'{optCritCmp12.numpy():10.3E},'\
               f'{optCritCmp22.numpy():10.3E} |'\
               f'{epochNr:>8d},'\
               f'{trainNr:>8d}/'\
               f'{len(trainList):<8d}'
      print(printStr)
    
    myoptSum.add(myopt)
    myoptN=myoptN+1
    if myoptN%10000==0:
      myoptN=0
      myoptSum.c1=0.0
      myoptSum.c2=0.0
      myoptSum.c3=0.0
    
    if False and (trainNr)%args.trainSaveInterval==0 and not restart:
      if args.train:
        exec(open(saveScript).read(), globals())
        if args.execRollout:
          for randSequence in args.evalSequence:
            #randSequence=args.plotSequence
            predictTimesteps=args.evalTimesteps
            writeVTK=False
            exec(open(rolloutScript).read(), globals())
    
    if breakComputation:
      break
  if breakComputation:
    breakComputation=False
    break
  if args.train:
    exec(open(saveScript).read(), globals())
    if args.execRollout:
      for randSequence in args.evalSequence:
        #randSequence=args.plotSequence
        predictTimesteps=args.evalTimesteps
        writeVTK=True
        exec(open(rolloutScript).read(), globals())

if restart:
  exec(open(loadScript).read(), globals())
  restart=False
  breakComputation=restart

if not args.train:
  print("Predicting rollout sequence in predict mode...")
  for randSequence in args.plotSequence:
    #randSequence=args.plotSequence
    predictTimesteps=args.predictTimesteps
    writeVTK=True
    exec(open(rolloutScript).read(), globals())
