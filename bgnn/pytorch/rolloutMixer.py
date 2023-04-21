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

lockStat=True

print(randSequence)
startTime=offset+(rp["nrPastVelocities"]+1)*rolloutStepSize

if rolloutEval:
  recordTimesteps=2**np.arange(20)
  #recordTimesteps=recordTimesteps[recordTimesteps>=rp["nrPastVelocities"]+1]
  recordTimesteps=recordTimesteps+rp["nrPastVelocities"]
  nextRecordInd=0
  recordList=[]

#xSceneData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xSceneDataList[randSequence].shape[1], 3, 3), dtype=np.float32)
#xSceneData[0:(rp["nrPastVelocities"])]=xSceneDataList[randSequence][(startTime-((rp["nrPastVelocities"]+1)*rolloutStepSize)):(startTime-(1*rolloutStepSize)):rolloutStepSize]
xParticleData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1], 3), dtype=np.float32)
xParticleData[0:(rp["nrPastVelocities"]+1)]=xParticleDataList[randSequence][(startTime-((rp["nrPastVelocities"]+1)*rolloutStepSize)):(startTime):rolloutStepSize]
maxParticleNoiseData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1]), dtype=np.float32)
maxParticleNoiseData[0:(rp["nrPastVelocities"]+1)]=maxParticleNoiseDataList[randSequence][(startTime-((rp["nrPastVelocities"]+1)*rolloutStepSize)):(startTime):rolloutStepSize]
maxParticleCoordNoisePosData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1],3), dtype=np.float32)
maxParticleCoordNoisePosData[0:(rp["nrPastVelocities"]+1)]=maxParticleCoordNoisePosDataList[randSequence][(startTime-((rp["nrPastVelocities"]+1)*rolloutStepSize)):(startTime):rolloutStepSize]
maxParticleCoordNoiseNegData=np.zeros((predictTimesteps+rp["nrPastVelocities"]+1, xParticleDataList[randSequence].shape[1],3), dtype=np.float32)
maxParticleCoordNoiseNegData[0:(rp["nrPastVelocities"]+1)]=maxParticleCoordNoiseNegDataList[randSequence][(startTime-((rp["nrPastVelocities"]+1)*rolloutStepSize)):(startTime):rolloutStepSize]

timestep=rp["nrPastVelocities"]
for timestep in range(rp["nrPastVelocities"]+1, predictTimesteps+rp["nrPastVelocities"]+1):
  #xSceneData[timestep-1,:]=np.matmul(myMoveRot, xSceneData[timestep-2,:].swapaxes(0,2).reshape(3,-1)).reshape(3,3,-1).swapaxes(0,2)
  if rolloutEval or vtkOutput or npOutput:
    print(f"Predicting Timestp: {timestep}", end="\r")
  nrParticlesList=[]
  nextPositionVecList=[]
  currentPositionVecList=[]
  currentVelocityVecList=[]
  nextPositionVecPerturbatedList=[]
  currentPositionVecPerturbatedList=[]
  currentVelocityVecPerturbatedList=[]
  
  batch=[]
  nrParticles=xParticleData.shape[1]
  #nrConstraints=xSceneData.shape[1]
  nrConstraints=sum([x.vectors.shape[0] for x in [blade1, blade4, mixer1, mixer4, shaft, mixingDrum]])
  
  
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
  xSceneDataPrev=rotMesh
  
  #currentConstPositionVec=torch.from_numpy(np.float32(xSceneData[timestep-1,:])).to(device)
  currentConstPositionVec=torch.from_numpy(np.float32(xSceneDataPrev)).to(device)
  
  prevPositionVec=torch.from_numpy(np.float32(xParticleData[timestep-2,:])).to(device)
  currentPositionVec=torch.from_numpy(np.float32(xParticleData[timestep-1,:])).to(device)
  nextPositionVec=torch.from_numpy(np.float32(xParticleData[timestep,:])).to(device)
  recentPositionVecs=torch.from_numpy(np.float32(xParticleData[(timestep-(rp["nrPastVelocities"]+1)):(timestep)])).to(device)
  
  recentMaxParticleNoise=torch.zeros((rp["nrPastVelocities"]+1, nrParticles), device=device)
  nextMaxParticleNoise=torch.zeros((1,nrParticles), device=device)
  recentMaxParticleNoisePos=torch.zeros((rp["nrPastVelocities"]+1, nrParticles, 3), device=device)
  nextMaxParticleNoisePos=torch.zeros((1,nrParticles,3), device=device)
  recentMaxParticleNoiseNeg=torch.zeros((rp["nrPastVelocities"]+1, nrParticles, 3), device=device)
  nextMaxParticleNoiseNeg=torch.zeros((1,nrParticles,3), device=device)
  
  velocityFeatureVecs=recentPositionVecs[1:]-recentPositionVecs[:-1]
  currentVelocityVec=recentPositionVecs[-1]-recentPositionVecs[-2]
  currentVelocityLen=torch.reshape(torch.sqrt(torch.sum(currentVelocityVec**2, 1)), (-1,1))
  currentAccelerationVec=(nextPositionVec-recentPositionVecs[-1])-velocityFeatureVecs[-1]
  #currentAccelerationVec=nextPositionVec-2.0*recentPositionVecs[-1]+recentPositionVecs[-2]
  currentAccelerationLen=torch.reshape(torch.sqrt(torch.sum(currentAccelerationVec**2, 1)), (-1,1))
  
  randomNoise=torch.cat([torch.zeros((1, nrParticles, 3), device=device), myRandNorm((rp["nrPastVelocities"], nrParticles, 3), mean=[0.,0.,0.], stddev=np.array(rp["velNoise"])/(rp["nrPastVelocities"] ** 0.5), device=device)], 0)*0.0

  if rp["noisyTarget"]:
    randomNoise=torch.cat([randomNoise, randomNoise[-1:]],0)
  else:
    randomNoise=torch.cat([randomNoise, randomNoise[-1:]*0.0],0)
  
  maxNoise=torch.cat([recentMaxParticleNoise, nextMaxParticleNoise],0)
  maxNoisePos=torch.cat([recentMaxParticleNoisePos, nextMaxParticleNoisePos],0)
  maxNoiseNeg=torch.cat([recentMaxParticleNoiseNeg, nextMaxParticleNoiseNeg],0)
  noiseStrength=torch.sqrt(torch.sum(randomNoise**2,2))
  if rp["correctNoise"]==1:
    #all noise
    randomNoise=torch.where(torch.logical_and(randomNoise>maxNoisePos, torch.unsqueeze(noiseStrength>maxNoise, -1)), myRandUniform(maxNoisePos.shape, minval=0, maxval=maxNoisePos, device=device), randomNoise)
    randomNoise=torch.where(torch.logical_and(randomNoise<maxNoiseNeg, torch.unsqueeze(noiseStrength>maxNoise, -1)), myRandUniform(maxNoiseNeg.shape, minval=maxNoiseNeg, maxval=0, device=device), randomNoise)
  elif rp["correctNoise"]==2:
    #only feature noise, no target noise
    randomNoiseCp=torch.where(torch.logical_and(randomNoise>maxNoisePos, torch.unsqueeze(noiseStrength>maxNoise, -1)), myRandUniform(maxNoisePos.shape, minval=0, maxval=maxNoisePos, device=device), randomNoise)
    randomNoiseCp=torch.where(torch.logical_and(randomNoiseCp<maxNoiseNeg, torch.unsqueeze(noiseStrength>maxNoise, -1)), myRandUniform(maxNoiseNeg.shape, minval=maxNoiseNeg, maxval=0, device=device), randomNoiseCp)
    randomNoise=torch.cat([randomNoise[0:6], randomNoiseCp[6:]],0)
  elif rp["correctNoise"]==3:
    #only target noise + current noise correction, no other features noise correction
    randomNoiseCp=torch.where(torch.logical_and(randomNoise>maxNoisePos, torch.unsqueeze(noiseStrength>maxNoise, -1)), myRandUniform(maxNoisePos.shape, minval=0, maxval=maxNoisePos, device=device), randomNoise)
    randomNoiseCp=torch.where(torch.logical_and(randomNoiseCp<maxNoiseNeg, torch.unsqueeze(noiseStrength>maxNoise, -1)), myRandUniform(maxNoiseNeg.shape, minval=maxNoiseNeg, maxval=0, device=device), randomNoiseCp)
    randomNoise=torch.cat([randomNoise[0:5], randomNoiseCp[5:]],0)
  
  prevPositionVecPerturbated=prevPositionVec+randomNoise[-3]
  currentPositionVecPerturbated=currentPositionVec+randomNoise[-2]
  nextPositionVecPerturbated=nextPositionVec+randomNoise[-1]
  recentPositionVecsPerturbated=recentPositionVecs+randomNoise[0:-1]
  
  velocityFeatureVecsPerturbated=recentPositionVecsPerturbated[1:]-recentPositionVecsPerturbated[:-1]
  currentVelocityVecPerturbated=recentPositionVecsPerturbated[-1]-recentPositionVecsPerturbated[-2]
  currentVelocityLenPerturbated=torch.reshape(torch.sqrt(torch.sum(currentVelocityVecPerturbated**2, 1)), (-1,1))
  currentAccelerationVecPerturbated=(nextPositionVecPerturbated-recentPositionVecsPerturbated[-1])-velocityFeatureVecsPerturbated[-1]
  #currentAccelerationVecPerturbated=nextPositionVecPerturbated-2.0*recentPositionVecsPerturbated[-1]+recentPositionVecsPerturbated[-2]
  currentAccelerationLenPerturbated=torch.reshape(torch.sqrt(torch.sum(currentAccelerationVecPerturbated**2, 1)), (-1,1))
  
  if rp["individualCutoff"]:
    if rp["noisyGraph"]:
      velStrength=torch.sqrt(torch.sum(currentVelocityVecPerturbated**2,1))
    else:
      velStrength=torch.sqrt(torch.sum(currentVelocityVec**2,1))
    useCutoff=rp["neighborCutoff"]+velStrength
  else:
    useCutoff=torch.as_tensor(rp["neighborCutoff"], device=device).float()
  
  points=currentPositionVecPerturbated
  tcoord=currentConstPositionVec
  exec(open(wallInfoScript).read(), globals())
  srInd_1=takeDist[:,0].cpu().numpy()
  #srInd_2=(takeDist[:,1]+nrParticles).numpy()
  srInd_2=np.arange(len(srInd_1))+nrParticles
  wallParticleSenders=np.concatenate([srInd_2])
  wallParticleReceivers=np.concatenate([srInd_1])
  #uwcount=tf.unique_with_counts(srInd_1)
  #particleWallWeight=1.0/uwcount[2].numpy()[uwcount[1].numpy()].astype(np.float32)
  if rp["wallWeight"]==0:
    particleWallWeight=torch.ones([len(srInd_1), 1])
  elif rp["wallWeight"]==1:
    particleWallWeight=1.0/torch.gather(torch.zeros([nrParticles], device=device).scatter_add(0, takeDist[:,0], torch.ones([len(takeDist[:,0])], device=device)), 0, takeDist[:,0])
    particleWallWeight=torch.reshape(particleWallWeight, (-1,1))
  nrWallParticles=len(srInd_1)
  nrWallParticleDistances=len(wallParticleSenders)
  
  if rp["implementation"]==1:
    if rp["noisyGraph"]:
      myTree=scipy.spatial.cKDTree(currentPositionVecPerturbated.cpu().numpy())
      recvTree=myTree.query_ball_point(currentPositionVecPerturbated.cpu().numpy(), useCutoff.cpu().numpy())
    else:
      myTree=scipy.spatial.cKDTree(currentPositionVec.cpu().numpy())
      recvTree=myTree.query_ball_point(currentPositionVec.cpu().numpy(), useCutoff.cpu().numpy())
    sendTree=[np.repeat(i, len(recvTree[i])) for i in range(len(recvTree))]
    interParticleReceivers=np.concatenate(recvTree)
    interParticleSenders=np.concatenate(sendTree)
  elif rp["implementation"]==2:
    if rp["noisyGraph"]:
      myTree=sklearn.neighbors.KDTree(currentPositionVecPerturbated.cpu().numpy())
      recvTree=myTree.query_radius(currentPositionVecPerturbated.cpu().numpy(), r=useCutoff.cpu().numpy())
    else:
      myTree=sklearn.neighbors.KDTree(currentPositionVec.cpu().numpy())
      recvTree=myTree.query_radius(currentPositionVec.cpu().numpy(), r=useCutoff.cpu().numpy())
    sendTree=[np.repeat(i, len(recvTree[i])) for i in range(len(recvTree))]
    interParticleReceivers=np.concatenate(recvTree)
    interParticleSenders=np.concatenate(sendTree)
  elif rp["implementation"]==3:
    if rp["noisyGraph"]:
      myTree=scipy.spatial.cKDTree(torch.reshape(torch.cat([currentPositionVecPerturbated, currentPositionVecPerturbated+currentVelocityVecPerturbated],1), (-1,3)).cpu().numpy())
      recvTree=myTree.query_ball_point(currentPositionVecPerturbated.cpu().numpy(), useCutoff.cpu().numpy())
    else:
      myTree=scipy.spatial.cKDTree(torch.reshape(torch.cat([currentPositionVec, currentPositionVec+currentVelocityVec],1), (-1,3)).cpu().numpy())
      recvTree=myTree.query_ball_point(currentPositionVec.cpu().numpy(), useCutoff.cpu().numpy())
    recvTree=[[y//2 for y in x] for x in recvTree]
    recvTree=[[x[0]]+[x[i] for i in range(1,len(x)) if x[i]!=x[i-1]] for x in recvTree]
    sendTree=[np.repeat(i, len(recvTree[i])) for i in range(len(recvTree))]
    interParticleReceivers=np.concatenate(recvTree)
    interParticleSenders=np.concatenate(sendTree)
  elif rp["implementation"]==4:
    if len(useCutoff.shape)==0:
      useCutoff=useCutoff.reshape(1)
    useCutoff=torch.reshape(useCutoff,(-1,))
    
    matTake=torch.where(torch.sqrt(torch.sum((currentPositionVecPerturbated.unsqueeze(0)-currentPositionVecPerturbated.unsqueeze(1))**2,2))<0.5*(useCutoff.unsqueeze(0)+useCutoff.unsqueeze(1)))
    interParticleReceivers=matTake[0]
    interParticleSenders=matTake[1]
    interParticleReceivers=interParticleReceivers.cpu().numpy()
    interParticleSenders=interParticleSenders.cpu().numpy()
  
  if not rp["useSelfLoops"]:
    commMask=interParticleReceivers!=interParticleSenders
    interParticleReceivers=interParticleReceivers[commMask]
    interParticleSenders=interParticleSenders[commMask]
  nrParticleDistances=len(interParticleSenders)
  
  interParticleReceivers=torch.from_numpy(interParticleReceivers).to(device)
  interParticleSenders=torch.from_numpy(interParticleSenders).to(device)
  wallParticleReceivers=torch.from_numpy(wallParticleReceivers).to(device)
  wallParticleSenders=torch.from_numpy(wallParticleSenders).to(device)
  
  
  
  partDistVecPerturbated=-(currentPositionVecPerturbated[interParticleSenders]-currentPositionVecPerturbated[interParticleReceivers])
  partDistLenPerturbated=torch.reshape(torch.sqrt(torch.sum(partDistVecPerturbated**2, 1)), (-1,1))
  partUnitDistVecPerturbated=partDistVecPerturbated/partDistLenPerturbated
  partUnitDistVecPerturbated=torch.where(torch.isfinite(partUnitDistVecPerturbated), partUnitDistVecPerturbated, torch.as_tensor(0.0, device=device).float())
  
  wallDistVec=-diffVec[takeDist[:,0], takeDist[:,1]]
  wallDistLen=torch.sqrt(torch.reshape(torch.sum(wallDistVec**2,1),(-1,1)))
  wallUnitDistVec=wallDistVec/wallDistLen
  wallUnitDistVec=torch.where(torch.isfinite(wallUnitDistVec), wallUnitDistVec, torch.as_tensor(0.0, device=device).float())
  
  projectedPartDistVecSendersPerturbated=torch.reshape(torch.sum(currentVelocityVecPerturbated[interParticleSenders]*partUnitDistVecPerturbated,1),(-1,1))*partUnitDistVecPerturbated
  projectedPartDistLenSendersPerturbated=torch.sqrt(torch.sum(projectedPartDistVecSendersPerturbated**2,1, keepdims=True))
  projectedPartDistVecReceiversPerturbated=torch.reshape(torch.sum(currentVelocityVecPerturbated[interParticleReceivers]*partUnitDistVecPerturbated,1),(-1,1))*partUnitDistVecPerturbated
  projectedPartDistLenReceiversPerturbated=torch.sqrt(torch.sum(projectedPartDistVecReceiversPerturbated**2,1, keepdims=True))
  projectedPartDistVecSumPerturbated=projectedPartDistVecSendersPerturbated-projectedPartDistVecReceiversPerturbated
  projectedPartDistLenSumPerturbated=torch.sqrt(torch.sum(projectedPartDistVecSumPerturbated**2,1, keepdims=True))
  
  projectedWallDistVecPerturbated=torch.reshape(torch.sum(currentVelocityVecPerturbated[takeDist[:,0]]*wallUnitDistVec,1),(-1,1))*wallUnitDistVec
  projectedWallDistLenPerturbated=torch.sqrt(torch.sum(projectedWallDistVecPerturbated**2,1, keepdims=True))
  
  partDistVecVMod=(partDistVecPerturbated-projectedPartDistVecSumPerturbated)
  partDistLenVMod=torch.reshape(torch.sqrt(torch.sum(partDistVecVMod**2, 1)), (-1,1))
  
  wallDistVecVMod=(wallDistVec-projectedWallDistVecPerturbated)
  wallDistLenVMod=torch.sqrt(torch.reshape(torch.sum(wallDistVec**2,1),(-1,1)))
  
  #angle=torch.abs(torch.sum(currentVelocityVecPerturbated[takeDist[:,0]]*nvec[takeDist[:,1]],1)/(torch.sqrt(torch.sum((currentVelocityVecPerturbated[takeDist[:,0]])**2,1))*torch.sqrt(torch.sum((nvec[takeDist[:,1]])**2,1))))
  #angle=torch.where(torch.isnan(angle), torch.as_tensor(0.0, device=device).float(), angle)
  #angle=angle*rp["multAngle"]
  
  #ucurrentConstPositionVec=torch.unique(torch.reshape(currentConstPositionVec,(-1,3)), dim=0)
  #devDistVec=torch.reshape(ucurrentConstPositionVec,(-1,1,3))-torch.reshape(ucurrentConstPositionVec,(1,-1,3))
  #devDistLen=torch.sqrt(torch.sum(devDistVec**2, 2, keepdims=True))
  #devDistMax=0.5*torch.max(devDistLen)
  ##devDistVec=tf.reshape(ucurrentConstPositionVec,(1,-1,3))-tf.reshape(tf.reduce_mean(currentConstPositionVec, 1),(-1,1,3))
  ##devDistLen=tf.sqrt(tf.reduce_sum(devDistVec**2, 2, keepdims=True))
  ##partDevDistVec=tf.reshape(ucurrentConstPositionVec,(1,-1,3))-tf.reshape(points,(-1,1,3))
  ##partDevDistLen=tf.sqrt(tf.reduce_sum(partDevDistVec**2, 2, keepdims=True))
  ##devDistLen=tf.gather(devDistLen, takeDist[:,1])
  ##devDistVec=tf.gather(devDistVec, takeDist[:,1])
  #partDevDistVec=torch.reshape(points,(-1,1,3))-torch.reshape(ucurrentConstPositionVec,(1,-1,3))
  #partDevDistLen=torch.sqrt(torch.sum(partDevDistVec**2, 2, keepdims=True))
  #wallPoints=minPoint[takeDist[:,0], takeDist[:,1]]
  #devDistVec=torch.reshape(wallPoints,(-1,1,3))-torch.reshape(ucurrentConstPositionVec,(1,-1,3))
  #devDistLen=torch.sqrt(torch.sum(devDistVec**2, 2, keepdims=True))
  
  
  
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
  currentAccelerationVecNormW=(torch.zeros((nrWallParticles, 3), device=device)-meanAccW)/stdAccW
  
  if rp["gravBias"]:
    currentAccelerationVecNormP=currentAccelerationVecNormP-torch.from_numpy(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32)).to(device)/stdAccP
    currentAccelerationVecPerturbatedNormP=currentAccelerationVecPerturbatedNormP-torch.from_numpy(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32)).to(device)/stdAccP
    currentAccelerationVecNormW=currentAccelerationVecNormW-torch.from_numpy(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32)).to(device)/stdAccW
  
  
  exec(open(featureOptionsScript).read(), globals())
  
  
  if len(particleNodeDataList)<2:
    particleNodeDataList.append(torch.ones((nrParticles,1)))
    particleNodeDataList.append(-torch.ones((nrParticles,1)))
    wallNodeDataList.append( torch.ones((nrWallParticles,1)))
    wallNodeDataList.append(-torch.ones((nrWallParticles,1)))
  
  #for particle embedding
  #particleNodeDataList.append(torch.from_numpy(typeParticleDataList[randSequence][timestep-1*rp["stepSize"]]-1).float().to(device).reshape(nrParticles,1))
  particleNodeDataList.append(torch.full((nrParticles,1), 1.0, device=device, dtype=torch.float32))
  wallNodeDataList.append(torch.full((nrWallParticles,1), 0.0, device=device, dtype=torch.float32))
  
  
  
  
  if len(particleEdgeDataList)<2:
    particleEdgeDataList.append(torch.ones((nrParticleDistances,1)))
    particleEdgeDataList.append(-torch.ones((nrParticleDistances,1)))
    wallEdgeDataList.append( torch.ones((nrWallParticleDistances,1)))
    wallEdgeDataList.append(-torch.ones((nrWallParticleDistances,1)))
  
  particleNodeData=torch.cat(particleNodeDataList, 1)
  wallNodeData=torch.cat(wallNodeDataList, 1)
  particleEdgeData=torch.cat(particleEdgeDataList, 1)
  wallEdgeData=torch.cat(wallEdgeDataList, 1)
  
  nodeFeatures=torch.cat([particleNodeData, wallNodeData], 0)
  edgeFeatures=torch.cat([particleEdgeData, wallEdgeData], 0)
  senders=torch.cat([interParticleSenders, wallParticleSenders])
  receivers=torch.cat([interParticleReceivers, wallParticleReceivers])
  edge_index=torch.stack([senders, receivers])
  
  
  
  if rp["wallOptImp"]>0.0:
    importance=rp["wallOptImp"]*torch.zeros([nrParticles], device=device).scatter_add(0, takeDist[:,0], (bstat.getStd("wallDistLen")[0]/(wallDistLen[:,0]+0.01*bstat.getStd("wallDistLen")[0])).float()).reshape((-1,1))
    targetFeaturesNorm=torch.cat([torch.cat([currentAccelerationVecNormP, currentAccelerationVecNormW], 0), 
                                  torch.cat([torch.ones((nrParticles,1), device=device)+importance, torch.zeros((nrWallParticles,1), device=device)], 0)
                                  ], 1)
    targetFeaturesPerturbatedNorm=torch.cat([torch.cat([currentAccelerationVecPerturbatedNormP, currentAccelerationVecNormW], 0), 
                                             torch.cat([torch.ones((nrParticles,1), device=device)+importance, torch.zeros((nrWallParticles,1), device=device)], 0)
                                             ], 1)
  else:
    targetFeaturesNorm=torch.cat([torch.cat([currentAccelerationVecNormP, currentAccelerationVecNormW], 0), 
                                  torch.cat([torch.ones((nrParticles,1), device=device), torch.zeros((nrWallParticles,1), device=device)], 0)
                                  ], 1)
    targetFeaturesPerturbatedNorm=torch.cat([torch.cat([currentAccelerationVecPerturbatedNormP, currentAccelerationVecNormW], 0), 
                                             torch.cat([torch.ones((nrParticles,1), device=device), torch.zeros((nrWallParticles,1), device=device)], 0)
                                             ], 1)
  
  nrParticlesList.append(nrParticles)
  nextPositionVecList.append(nextPositionVec)
  currentPositionVecList.append(currentPositionVec)
  currentVelocityVecList.append(currentVelocityVec)
  nextPositionVecPerturbatedList.append(nextPositionVecPerturbated)
  currentPositionVecPerturbatedList.append(currentPositionVecPerturbated)
  currentVelocityVecPerturbatedList.append(currentVelocityVecPerturbated)
  
  #targetFeaturesNormDictList.append(targetFeaturesNormDict)
  #targetFeaturesPerturbatedNormDictList.append(targetFeaturesPerturbatedNormDict)
  if rp["noisyTarget"]:
    batch.append(torch_geometric.data.Data(x=nodeFeatures, edge_index=edge_index, edge_attr=edgeFeatures, y=targetFeaturesPerturbatedNorm))
  else:
    batch.append(torch_geometric.data.Data(x=nodeFeatures, edge_index=edge_index, edge_attr=edgeFeatures, y=targetFeaturesNorm))
  
  batchInd=0
  nrParticles=nrParticlesList[batchInd]
  nextPositionVec=nextPositionVecList[batchInd]
  currentPositionVec=currentPositionVecList[batchInd]
  currentVelocityVec=currentVelocityVecList[batchInd]
  nextPositionVecPerturbated=nextPositionVecPerturbatedList[batchInd]
  currentPositionVecPerturbated=currentPositionVecPerturbatedList[batchInd]
  currentVelocityVecPerturbated=currentVelocityVecPerturbatedList[batchInd]
  with torch.no_grad():
    myres=useGN(edge_index=batch[0].edge_index, nodeFeatures=batch[0].x, edgeFeatures=batch[0].edge_attr)
  if rp["constrainPredictionLength"]==1:
    myres=torch.stack([(myres[:,0])*rp["neighborCutoff"]*torch.sin(torch.sigmoid(myres[:,1])*np.pi*1.5-np.pi*0.25)*torch.cos(torch.sigmoid(myres[:,2])*np.pi*2.0*1.5-np.pi*2.0*0.25),
    (myres[:,0])*rp["neighborCutoff"]*torch.sin(torch.sigmoid(myres[:,1])*np.pi*1.5-np.pi*0.25)*torch.sin(torch.sigmoid(myres[:,2])*np.pi*2.0*1.5-np.pi*2.0*0.25),
    (myres[:,0])*rp["neighborCutoff"]*torch.cos(torch.sigmoid(myres[:,1])*np.pi*1.5-np.pi*0.25)]).T
  elif rp["constrainPredictionLength"]==2:
    normVector=torch.sqrt(torch.reshape(torch.sum(myres[:,0:3]**2,1),(-1,1)))
    dirVector=myres[:,0:3]/normVector
    myres=dirVector*(myres[:,3:4])*rp["neighborCutoff"]
  elif rp["constrainPredictionLength"]==3:
    normVector=torch.sqrt(torch.reshape(torch.sum(myres[:,0:3]**2,1),(-1,1)))
    dirVector=myres[:,0:3]/normVector
    myres=dirVector*(myres[:,3:4])*rp["neighborCutoff"]
  if False: #rp["gravBias"]:
    predAcceleration=(myres[:nrParticles]*stdAccP)+meanAccP+torch.from_numpy(np.array([[0.0, 0.0,-9.81*(float(rp["stepSize"])*2.5e-5)**2]], dtype=np.float32)).to(device)
  else:
    predAcceleration=(myres[:nrParticles]*stdAccP)+meanAccP
  
  if args.accScale:
    accStrength=torch.sqrt(torch.reshape(torch.sum(predAcceleration**2,1),(-1,1)))
    predAcceleration=torch.where(accStrength>rp["neighborCutoff"], rp["neighborCutoff"]*(predAcceleration/accStrength), predAcceleration)
  
  predVelocity=currentVelocityVec+predAcceleration
  predPosition=currentPositionVec+predVelocity
  xParticleData[timestep,:]=predPosition.detach().cpu().numpy()
  maxParticleNoiseData[timestep,:]=maxNoise[-1].cpu().numpy()
  maxParticleCoordNoisePosData[timestep,:]=maxNoisePos[-1].cpu().numpy()
  maxParticleCoordNoiseNegData[timestep,:]=maxNoiseNeg[-1].cpu().numpy()
  
  if rolloutEval and timestep==recordTimesteps[nextRecordInd] and len(xParticleDataList[randSequence])>timestep:
    optMat=ot.dist(xParticleData[timestep,:], np.array(xParticleDataList[randSequence][timestep,:]))
    recordList.append(np.sum(optMat*ot.emd(ot.unif(optMat.shape[0]), ot.unif(optMat.shape[1]), M=optMat, numItermax=1000000)))
    nextRecordInd=nextRecordInd+1

if rolloutEval or vtkOutput or npOutput:
  print()
timestep=timestep+1
#xSceneData[timestep-1,:]=np.matmul(myMoveRot, xSceneData[timestep-2,:].swapaxes(0,2).reshape(3,-1)).reshape(3,3,-1).swapaxes(0,2)
#myiter=timestep-1
#angle=(360.0/((rpMain/computeTimestep)/(dumpTimeStep/computeTimestep)))
#rotMatIter=np.array([[1.0, 0.0, 0.0], [0.0, np.cos(((angle*myiter)*math.pi)/180), -np.sin(((angle*myiter)*math.pi)/180)], [0.0, np.sin(((angle*myiter)*math.pi)/180), np.cos(((angle*myiter)*math.pi)/180)]])
#rotBlade1=np.matmul(blade1.vectors, rotMatIter.T)
#rotBlade4=np.matmul(blade4.vectors, rotMatIter.T)
#rotMixer1=np.matmul(mixer1.vectors, rotMatIter.T)
#rotMixer4=np.matmul(mixer4.vectors, rotMatIter.T)
#rotShaft=np.matmul(shaft.vectors, rotMatIter.T)
#rotMixingDrum=mixingDrum.vectors
#rotMesh=np.vstack([rotShaft, rotBlade1, rotBlade4, rotMixer1, rotMixer4, rotMixingDrum])
#xSceneData=rotMesh



if rolloutEval:
  emdRes=np.array(recordList)
  for indTime in range(0, len(recordList)):
    timepoint=(recordTimesteps[:len(recordList)])[indTime]
  
  if emdOutput:
    evaluationsFilePrefix=os.path.join("/system/user/mayr-data/BGNNRuns/evaluations", rp["problem"], args.experiment, saveFilePrefix.split("/")[-2])
    if not os.path.exists(os.path.join(evaluationsFilePrefix)):
      os.makedirs(evaluationsFilePrefix)
    f=open(os.path.join(evaluationsFilePrefix, "emd_"+str(randSequence)+"_"+str(startTime)+".pckl"), "wb")
    pickle.dump(emdRes, f, -1)
    f.close()



if vtkOutput:
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
      
      
      
      #xDataScene=xSceneData[ind,:]
      ##xDataScene=triangleCoord_Data
      #points=vtk.vtkPoints()
      #vertices=vtk.vtkCellArray()
      #for i in range(0, xDataScene.shape[0]):
      #  vertices.InsertNextCell(3)
      #  for vind in range(0, xDataScene.shape[1]):
      #    ins=points.InsertNextPoint(xDataScene[i][vind])
      #    vertices.InsertCellPoint(ins)
      
      #polydata=vtk.vtkPolyData()
      #polydata.SetPoints(points)
      #polydata.SetPolys(vertices)
      
      #writer=vtk.vtkXMLPolyDataWriter()
      #writer.SetFileName(os.path.join(rolloutFilePrefix, "wall_"+str((ind*rp["stepSize"])+1)+".vtp"))
      ##writer.SetFileName("/system/user/mayr/test.vtp")
      #writer.SetInputData(polydata)
      #writer.SetDataModeToBinary()
      #writer.SetDataModeToAscii()
      #writer.Write()
      
      
      
  print()

if npOutput:
  rolloutDirPrefix=os.path.join("/system/user/mayr-data/BGNNRuns/trajectories", rp["problem"], args.experiment, saveFilePrefix.split("/")[-2])
  if not os.path.exists(os.path.join(rolloutDirPrefix)):
    os.makedirs(rolloutDirPrefix)
  
  saveFile=os.path.join(rolloutDirPrefix, "predp_"+str(randSequence)+"_"+str(startTime)+".npy")
  np.save(saveFile, xParticleData)
  
  #saveFile=os.path.join(rolloutDirPrefix, "preds_"+str(randSequence)+"_"+str(startTime)+".npy")
  #np.save(saveFile, xSceneData)
  
  saveFile=os.path.join(rolloutDirPrefix, "gtp_"+str(randSequence)+"_"+str(startTime)+".npy")
  gtxParticleData=xParticleDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime-(rp["nrPastVelocities"]+1)+xParticleData.shape[0])]
  np.save(saveFile, gtxParticleData)
  
  #saveFile=os.path.join(rolloutDirPrefix, "gts_"+str(randSequence)+"_"+str(startTime)+".npy")
  #gtxSceneData=xSceneDataList[randSequence][(startTime-(rp["nrPastVelocities"]+1)):(startTime-(rp["nrPastVelocities"]+1)+xSceneData.shape[0])]
  #np.save(saveFile, gtxSceneData)

lockStat=False