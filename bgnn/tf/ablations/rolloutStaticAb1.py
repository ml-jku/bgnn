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

data_output=1e-3
myMoveRot=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
lockStat=True

print(randSequence)
startTime=rp["nrPastVelocities"]+1

recordTimesteps=2**np.arange(20)
recordTimesteps=recordTimesteps+rp["nrPastVelocities"]
nextRecordInd=0
recordList=[]

xSceneData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xSceneDataList[randSequence].shape[1], 3, 3), dtype=np.float32)
xSceneData[0:(rp["nrPastVelocities"])]=xSceneDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime-1)]
xParticleData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1], 3), dtype=np.float32)
xParticleData[0:(rp["nrPastVelocities"]+1)]=xParticleDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime)]
maxParticleNoiseData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1]), dtype=np.float32)
maxParticleNoiseData[0:(rp["nrPastVelocities"]+1)]=maxParticleNoiseDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime)]
maxParticleCoordNoisePosData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1],3), dtype=np.float32)
maxParticleCoordNoisePosData[0:(rp["nrPastVelocities"]+1)]=maxParticleCoordNoisePosDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime)]
maxParticleCoordNoiseNegData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1],3), dtype=np.float32)
maxParticleCoordNoiseNegData[0:(rp["nrPastVelocities"]+1)]=maxParticleCoordNoiseNegDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime)]

for timestep in range(rp["nrPastVelocities"]+1, predictTimesteps+rp["nrPastVelocities"]+1):
  xSceneData[timestep-1,:]=np.matmul(myMoveRot, xSceneData[timestep-2,:].swapaxes(0,2).reshape(3,-1)).reshape(3,3,-1).swapaxes(0,2)
  print(f"Predicting Timestp: {timestep}", end="\r")
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
  
  nrParticles=xParticleData.shape[1]
  nrConstraints=xSceneData.shape[1]
  
  with tf.device('/device:GPU:0'):
    currentConstPositionVec=tf.identity(np.float32(xSceneData[timestep-1,:]))
    
    prevPositionVec=tf.identity(np.float32(xParticleData[timestep-2,:]))
    currentPositionVec=tf.identity(np.float32(xParticleData[timestep-1,:]))
    nextPositionVec=tf.identity(np.float32(xParticleData[timestep,:]))
    recentPositionVecs=tf.identity(np.float32(xParticleData[(timestep-(rp["nrPastVelocities"]+1)):(timestep)]))
    
    recentMaxParticleNoise=tf.zeros((rp["nrPastVelocities"]+1, nrParticles))
    nextMaxParticleNoise=tf.zeros((1,nrParticles))
    recentMaxParticleNoisePos=tf.zeros((rp["nrPastVelocities"]+1, nrParticles, 3))
    nextMaxParticleNoisePos=tf.zeros((1,nrParticles,3))
    recentMaxParticleNoiseNeg=tf.zeros((rp["nrPastVelocities"]+1, nrParticles, 3))  
    nextMaxParticleNoiseNeg=tf.zeros((1,nrParticles,3))
    
  velocityFeatureVecs=recentPositionVecs[1:]-recentPositionVecs[:-1]
  currentVelocityVec=recentPositionVecs[-1]-recentPositionVecs[-2]
  currentVelocityLen=tf.reshape(tf.sqrt(tf.reduce_sum(currentVelocityVec**2, 1)), (-1,1))
  currentAccelerationVec=(nextPositionVec-recentPositionVecs[-1])-velocityFeatureVecs[-1]
  #currentAccelerationVec=nextPositionVec-2.0*recentPositionVecs[-1]+recentPositionVecs[-2]
  currentAccelerationLen=tf.reshape(tf.sqrt(tf.reduce_sum(currentAccelerationVec**2, 1)), (-1,1))
  
  randomNoise=tf.concat([tf.zeros((1, nrParticles, 3)), tf.random.normal((rp["nrPastVelocities"], nrParticles, 3), mean=[0.,0.,0.], stddev=np.array(rp["velNoise"])/(rp["nrPastVelocities"] ** 0.5))], 0)*0.0

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
    #only feature noise, no target noise
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
  shuffleMat=tf.concat([takeDist, tf.stack([takeDist[:,0], takeDist[:,1]+diffVec.shape[1]],1)],0)
  shuffleMat=tf.concat([shuffleMat, tf.reshape(tf.identity(np.concatenate([srInd_2, srInd_1])),(-1,1)), tf.reshape(tf.identity(np.concatenate([srInd_1, srInd_2])),(-1,1))],1)
  shuffleMat=tf.random.shuffle(shuffleMat)
  diffVecEdge=tf.concat([diffVec, -diffVec], 1)
  distsEdge=tf.sqrt(tf.reduce_sum(diffVecEdge**2, 2))
  takeDistEdge=shuffleMat[:,0:2]
  wallParticleSenders=shuffleMat[:,2]
  wallParticleReceivers=shuffleMat[:,3]
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
  
  wallDistVec=-tf.gather_nd(diffVecEdge, takeDistEdge)
  wallDistLen=tf.sqrt(tf.reshape(tf.reduce_sum(wallDistVec**2,1),(-1,1)))
  wallUnitDistVec=wallDistVec/wallDistLen
  wallUnitDistVec=tf.where(tf.math.is_finite(wallUnitDistVec), wallUnitDistVec, 0)
  
  projectedPartDistVecSendersPerturbated=tf.reshape(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, interParticleSenders)*partUnitDistVecPerturbated,1),(-1,1))*partUnitDistVecPerturbated
  projectedPartDistLenSendersPerturbated=tf.sqrt(tf.reduce_sum(projectedPartDistVecSendersPerturbated**2,1, keepdims=True))
  projectedPartDistVecReceiversPerturbated=tf.reshape(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, interParticleReceivers)*partUnitDistVecPerturbated,1),(-1,1))*partUnitDistVecPerturbated
  projectedPartDistLenReceiversPerturbated=tf.sqrt(tf.reduce_sum(projectedPartDistVecReceiversPerturbated**2,1, keepdims=True))
  projectedPartDistVecSumPerturbated=projectedPartDistVecSendersPerturbated-projectedPartDistVecReceiversPerturbated
  projectedPartDistLenSumPerturbated=tf.sqrt(tf.reduce_sum(projectedPartDistVecSumPerturbated**2,1, keepdims=True))
  
  projectedWallDistVecPerturbated=tf.reshape(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, takeDistEdge[:,0])*wallUnitDistVec,1),(-1,1))*wallUnitDistVec
  projectedWallDistLenPerturbated=tf.sqrt(tf.reduce_sum(projectedWallDistVecPerturbated**2,1, keepdims=True))
  
  partDistVecVMod=(partDistVecPerturbated-projectedPartDistVecSumPerturbated)
  partDistLenVMod=tf.reshape(tf.sqrt(tf.reduce_sum(partDistVecVMod**2, 1)), (-1,1))
  
  wallDistVecVMod=(wallDistVec-projectedWallDistVecPerturbated)
  wallDistLenVMod=tf.sqrt(tf.reshape(tf.reduce_sum(wallDistVec**2,1),(-1,1)))
  
  angle=tf.abs(tf.reduce_sum(tf.gather(currentVelocityVecPerturbated, takeDistEdge[:,0])*tf.gather(nvec, takeDistEdge[:,1]),1)/(tf.sqrt(tf.reduce_sum((tf.gather(currentVelocityVecPerturbated, takeDistEdge[:,0]))**2,1))*tf.sqrt(tf.reduce_sum((tf.gather(nvec, takeDistEdge[:,1]))**2,1))))
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
    if rp["wallReceivers"]:
      impwh=0.5
    else:
      impwh=1.0
    importance=rp["wallOptImp"]*tf.reshape((tf.scatter_nd(takeDistEdge[:,0:1], (impwh*bstat.getStd("wallDistLen")[0])/(wallDistLen[:,0]+0.01*bstat.getStd("wallDistLen")[0]), shape=[nrParticles])), (-1,1))
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
  
  myresBatch=myGN(graphInputBatch)
  
  batchInd=0
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
  
  if args.accScale:
    accStrength=tf.sqrt(tf.reshape(tf.reduce_sum(predAcceleration**2,1),(-1,1)))
    predAcceleration=tf.where(accStrength>rp["neighborCutoff"], rp["neighborCutoff"]*(predAcceleration/accStrength), predAcceleration)
  
  predVelocity=currentVelocityVec+predAcceleration
  predPosition=currentPositionVec+predVelocity
  xParticleData[timestep,:]=predPosition.numpy()
  maxParticleNoiseData[timestep,:]=maxNoise[-1].numpy()
  maxParticleCoordNoisePosData[timestep,:]=maxNoisePos[-1].numpy()
  maxParticleCoordNoiseNegData[timestep,:]=maxNoiseNeg[-1].numpy()
  
  if timestep==recordTimesteps[nextRecordInd] and len(xParticleDataList[randSequence])>timestep:
    optMat=ot.dist(xParticleData[timestep,:], xParticleDataList[randSequence][timestep,:])
    recordList.append(np.sum(optMat*ot.emd(ot.unif(optMat.shape[0]), ot.unif(optMat.shape[1]), M=optMat)))
    nextRecordInd=nextRecordInd+1
  
print()


evaluationsFilePrefix=os.path.join("/system/user/mayr-data/BGNNRuns/evaluations", rp["problem"], args.experiment, saveFilePrefix.split("/")[-2])
if not os.path.exists(os.path.join(evaluationsFilePrefix)):
  os.makedirs(evaluationsFilePrefix)

emdRes=np.array(recordList)

f=open(os.path.join(evaluationsFilePrefix, "emd_"+str(randSequence)+"_"+str(startTime)+".pckl"), "wb")
pickle.dump(emdRes, f, -1)
f.close()

if args.train:
  for indTime in range(0, len(recordList)):
    timepoint=(recordTimesteps[:len(recordList)])[indTime]

if writeVTK:
  rolloutFilePrefix=os.path.join("/system/user/mayr-data/BGNNRuns/predictions", rp["problem"], args.experiment, saveFilePrefix.split("/")[-2], str(randSequence)+"_"+str(startTime))
  if not os.path.exists(os.path.join(rolloutFilePrefix)):
    os.makedirs(rolloutFilePrefix)
  
  
  
  typeData=typeParticleDataList[randSequence][0,:]
  radiusData=radiusParticleDataList[randSequence][0,:]
  
  tsModulo=args.modulo//rp["stepSize"]
  print(rolloutFilePrefix)
  for ind in range(len(xParticleData)):
    if ind%tsModulo==0:
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
      writer.SetFileName(os.path.join(rolloutFilePrefix, "pred_"+str((ind*rp["stepSize"])+1)+".vtp"))
      writer.SetInputData(polydata)
      writer.SetDataModeToBinary()
      writer.SetDataModeToAscii()
      writer.Write()
      
      
      
      xDataScene=xSceneData[ind,:]
      #xDataScene=triangleCoord_Data
      points=vtk.vtkPoints()
      vertices=vtk.vtkCellArray()
      for i in range(0, xDataScene.shape[0]):
        vertices.InsertNextCell(3)
        for vind in range(0, xDataScene.shape[1]):
          ins=points.InsertNextPoint(xDataScene[i][vind])
          vertices.InsertCellPoint(ins)
      
      polydata=vtk.vtkPolyData()
      polydata.SetPoints(points)
      polydata.SetPolys(vertices)
      
      writer=vtk.vtkXMLPolyDataWriter()
      writer.SetFileName(os.path.join(rolloutFilePrefix, "wall_"+str((ind*rp["stepSize"])+1)+".vtp"))
      #writer.SetFileName("/system/user/mayr/test.vtp")
      writer.SetInputData(polydata)
      writer.SetDataModeToBinary()
      writer.SetDataModeToAscii()
      writer.Write()
      
      
      
  print()

if args.npOutput:
  rolloutDirPrefix=os.path.join("/system/user/mayr-data/BGNNRuns/trajectories", rp["problem"], args.experiment, saveFilePrefix.split("/")[-2])
  if not os.path.exists(os.path.join(rolloutDirPrefix)):
    os.makedirs(rolloutDirPrefix)
  
  saveFile=os.path.join(rolloutDirPrefix, "predp_"+str(randSequence)+"_"+str(startTime)+".npy")
  np.save(saveFile, xParticleData)
  
  saveFile=os.path.join(rolloutDirPrefix, "preds_"+str(randSequence)+"_"+str(startTime)+".npy")
  np.save(saveFile, xSceneData)
  
  saveFile=os.path.join(rolloutDirPrefix, "gtp_"+str(randSequence)+"_"+str(startTime)+".npy")
  gtxParticleData=xParticleDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime-(rp["nrPastVelocities"]+1)+xParticleData.shape[0])]
  np.save(saveFile, gtxParticleData)
  
  saveFile=os.path.join(rolloutDirPrefix, "gts_"+str(randSequence)+"_"+str(startTime)+".npy")
  gtxSceneData=xSceneDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime-(rp["nrPastVelocities"]+1)+xSceneData.shape[0])]
  np.save(saveFile, gtxSceneData)
