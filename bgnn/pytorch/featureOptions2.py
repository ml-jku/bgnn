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

particleNodeDataList=[]
wallNodeDataList=[]
particleEdgeDataList=[]
wallEdgeDataList=[]
normreg=Normalize()


dbg=True
dbgInt=0
if rp["usePastVelocitiesVec"][0]:
  meanValP, stdValP, meanValW, stdValW=normParV(rp["usePastVelocitiesVec"][1], bstat, "currentVelocityVecPerturbated", "wallVelocityVec", "currentVelocityLenPerturbated", "wallVelocityLen", meanVelVec, stdVelLenPerturbated, stdVelVecPerturbated)
  upFeatures=torch.reshape(((velocityFeatureVecsPerturbated-meanValP)/stdValP).permute([1,0,2]), (nrParticles, -1))
  uwFeatures=torch.reshape(((torch.zeros((rp["nrPastVelocities"], nrWallParticles, 3), device=device)-meanValW)/stdValW).permute([1,0,2]), (nrWallParticles, rp["nrPastVelocities"]*3))
  
  particleNodeDataList.append(upFeatures)
  wallNodeDataList.append(uwFeatures)
  if dbg:
    dbgInt=1

if rp["usePastVelocitiesLen"][0]:
  meanValP, stdValP, meanValW, stdValW=normParS(rp["usePastVelocitiesLen"][1], bstat, "currentVelocityLenPerturbated", "wallVelocityLen", meanVelLen, stdVelLenPerturbated)
  upFeatures=torch.reshape(((torch.sqrt(torch.sum(velocityFeatureVecsPerturbated**2,2, keepdims=True))-meanValP)/stdValP).permute([1,0,2]), (nrParticles, -1))
  uwFeatures=torch.reshape(((torch.zeros((rp["nrPastVelocities"], nrWallParticles, 1), device=device)-meanValW)/stdValW).permute([1,0,2]), (nrWallParticles, rp["nrPastVelocities"]*1))
  
  particleNodeDataList.append(upFeatures)
  wallNodeDataList.append(uwFeatures)
  if dbg:
    dbgInt=2

if rp["usePastVelocitiesLen2"][0]:
  upFeatures=torch.sum(velocityFeatureVecsPerturbated**2,2, keepdims=True).permute([1,0,2])
  uwFeatures=torch.zeros((nrWallParticles, rp["nrPastVelocities"], 1), device=device)
  estat.trackE("pPastVelocities2", upFeatures, epochNr, lockStat)
  estat.trackE("wPastVelocities2", uwFeatures, epochNr, lockStat)
  
  defaultMean=np.array([[0.0]])
  defalutStd=stdVelLenPerturbated**2
  normreg.registerSN(normParS, rp["usePastVelocitiesLen2"][1], estat, "pPastVelocities2", "wPastVelocities2", defaultMean, defalutStd, len(particleNodeDataList), len(wallNodeDataList))
  particleNodeDataList.append(upFeatures)
  wallNodeDataList.append(uwFeatures)
  if dbg:
    dbgInt=3



#---



if rp["useWallDistLen"][0] or rp["useWallDistVec"][0]:
  upFeatures=dists
  uwFeatures=torch.zeros((nrWallParticles, nrConstraints), device=device)
  estat.trackE("pWallDist", upFeatures, epochNr, lockStat)
  estat.trackE("wWallDist", uwFeatures, epochNr, lockStat)
  
  if rp["useWallDistLen"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)*(1.0/rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/rp["neighborCutoff"])
    normreg.registerSN(normParS, rp["useWallDistLen"][1], estat, "pWallDist", "wWallDist", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=4
    
  if rp["useWallDistVec"][0]:
    upFeatures=diffVec
    uwFeatures=torch.zeros((nrWallParticles, nrConstraints, 3), device=device)
    estat.trackE("pWallDistVec", upFeatures, epochNr, lockStat)
    estat.trackE("wWallDistVec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(1.0/rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/rp["neighborCutoff"])
    normreg.registerVN(normParV, rp["useWallDistVec"][1], estat, "pWallDistVec", "wWallDistVec", "pWallDist", "wWallDist", defaultMean, defalutStdLen, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=5

if rp["useWallInvDistLen"][0] or rp["useWallInvDistVec"][0]:
  upFeatures=1.0/(0.001*rp["neighborCutoff"]+dists)
  uwFeatures=torch.zeros((nrWallParticles, nrConstraints), device=device)
  estat.trackE("pWallInvDist", upFeatures, epochNr, lockStat)
  estat.trackE("wWallInvDist", uwFeatures, epochNr, lockStat)
  
  if rp["useWallInvDistLen"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])
    normreg.registerSN(normParS, rp["useWallInvDistLen"][1], estat, "pWallInvDist", "wWallInvDist", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=6
  
  if rp["useWallInvDistVec"][0]:
    upFeatures=(diffVec/(0.001*rp["neighborCutoff"]+torch.unsqueeze(dists,2))**2)
    uwFeatures=torch.zeros((nrWallParticles, nrConstraints, 3), device=device)
    estat.trackE("pWallInvDistVec", upFeatures, epochNr, lockStat)
    estat.trackE("wWallInvDistVec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])
    normreg.registerVN(normParV, rp["useWallInvDistVec"][1], estat, "pWallInvDistVec", "wWallInvDistVec", "pWallInvDist", "wWallInvDist", defaultMean, defalutStdLen, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=7

if rp["useWallInvDist2Len"][0] or rp["useWallInvDist2Vec"][0]:
  upFeatures=1.0/(0.001*rp["neighborCutoff"]+dists)**2
  uwFeatures=(torch.zeros((nrWallParticles, nrConstraints), device=device))**2
  estat.trackE("pWallInvDist2", upFeatures, epochNr, lockStat)
  estat.trackE("wWallInvDist2", uwFeatures, epochNr, lockStat)
  
  if rp["useWallInvDist2Len"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"]**2)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"]**2)
    normreg.registerSN(normParS, rp["useWallInvDist2Len"][1], estat, "pWallInvDist2", "wWallInvDist2", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=8
  
  if rp["useWallInvDist2Vec"][0]:
    upFeatures=(diffVec/(0.001*rp["neighborCutoff"]+torch.unsqueeze(dists,2))**3)
    uwFeatures=torch.zeros((nrWallParticles, nrConstraints, 3), device=device)
    estat.trackE("pWallInvDist2Vec", upFeatures, epochNr, lockStat)
    estat.trackE("wWallInvDist2Vec", uwFeatures, epochNr, lockStat)

    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(rp["neighborCutoff"]**2)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"]**2)
    normreg.registerVN(normParV, rp["useWallInvDist2Vec"][1], estat, "pWallInvDist2Vec", "wWallInvDist2Vec", "pWallInvDist2", "wWallInvDist2", defaultMean, defalutStdLen, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=9

if rp["useWallDistLenClip"][0]:
  upFeatures=torch.clamp(dists/rp["neighborCutoff"], -1., 1.)
  uwFeatures=torch.zeros((nrWallParticles, nrConstraints), device=device)
  estat.trackE("pWallDistClip", upFeatures, epochNr, lockStat)
  estat.trackE("wWallDistClip", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)
  normreg.registerSN(normParS, rp["useWallDistLenClip"][1], estat, "pWallDistClip", "wWallDistClip", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
  particleNodeDataList.append(upFeatures)
  wallNodeDataList.append(uwFeatures)
  if dbg:
    dbgInt=10

if rp["useWallInvDistLenClip"][0]:
  upFeatures=torch.clamp(rp["neighborCutoff"]/(0.001*rp["neighborCutoff"]+dists), -1., 1.)
  uwFeatures=torch.zeros((nrWallParticles, nrConstraints), device=device)
  estat.trackE("pWallInvDistClip", upFeatures, epochNr, lockStat)
  estat.trackE("wWallInvDistClip", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)
  normreg.registerSN(normParS, rp["useWallInvDistLenClip"][1], estat, "pWallInvDistClip", "wWallInvDistClip", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
  particleNodeDataList.append(upFeatures)
  wallNodeDataList.append(uwFeatures)
  if dbg:
    dbgInt=11

if rp["useWallInvDistLen2Clip"][0]:
  upFeatures=torch.clamp(rp["neighborCutoff"]/(0.001*rp["neighborCutoff"]+dists)**2, -1., 1.)
  uwFeatures=(torch.zeros((nrWallParticles, nrConstraints), device=device))**2
  estat.trackE("pWallInvDist2Clip", upFeatures, epochNr, lockStat)
  estat.trackE("wWallInvDist2Clip", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)
  normreg.registerSN(normParS, rp["useWallInvDistLen2Clip"][1], estat, "pWallInvDist2Clip", "wWallInvDist2Clip", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
  particleNodeDataList.append(upFeatures)
  wallNodeDataList.append(uwFeatures)
  if dbg:
    dbgInt=12

if rp["useWallDistLenClipInv"][0]:
  upFeatures=(1-torch.clamp(dists/rp["neighborCutoff"], -1., 1.))
  uwFeatures=torch.ones((nrWallParticles, nrConstraints), device=device)
  estat.trackE("pWallDistClipInv", upFeatures, epochNr, lockStat)
  estat.trackE("wWallDistClipInv", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)
  normreg.registerSN(normParS, rp["useWallDistLenClipInv"][1], estat, "pWallDistClipInv", "wWallDistClipInv", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
  particleNodeDataList.append(upFeatures)
  wallNodeDataList.append(uwFeatures)
  if dbg:
    dbgInt=13



#---



if rp["useTPDistLen"][0] or rp["useTPDistVec"][0]:
  upFeatures=partDevDistLen
  uwFeatures=devDistLen
  estat.trackE("pTPDist", upFeatures, epochNr, lockStat)
  estat.trackE("wTPDist", uwFeatures, epochNr, lockStat)
  
  if rp["useTPDistLen"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/devDistMax)
    normreg.registerSN(normParS, rp["useTPDistLen"][1], estat, "pTPDist", "wTPDist", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=14
  
  if rp["useTPDistVec"][0]:
    upFeatures=partDevDistVec
    uwFeatures=devDistVec
    estat.trackE("pTPDistVec", upFeatures, epochNr, lockStat)
    estat.trackE("wTPDistVec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(1.0/devDistMax)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/devDistMax)
    normreg.registerVN(normParV, rp["useTPDistVec"][1], estat, "pTPDistVec", "wTPDistVec", "pTPDist", "wTPDist", defaultMean, defalutStdLen, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=15

if rp["useTPInvDistLen"][0] or rp["useTPInvDistVec"][0]:
  upFeatures=(1.0/(0.001*rp["neighborCutoff"]+partDevDistLen))
  uwFeatures=(1.0/(0.001*rp["neighborCutoff"]+devDistLen))
  estat.trackE("pTPInvDist", upFeatures, epochNr, lockStat)
  estat.trackE("wTPInvDist", uwFeatures, epochNr, lockStat)
  
  if rp["useTPInvDistLen"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(devDistMax)
    normreg.registerSN(normParS, rp["useTPInvDistLen"][1], estat, "pTPInvDist", "wTPInvDist", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=16
  
  if rp["useTPInvDistVec"][0]:
    upFeatures=(partDevDistVec/(0.001*rp["neighborCutoff"]+partDevDistLen)**2)
    uwFeatures=(devDistVec/(0.001*rp["neighborCutoff"]+devDistLen)**2)
    estat.trackE("pTPInvDistVec", upFeatures, epochNr, lockStat)
    estat.trackE("wTPInvDistVec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(devDistMax)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(devDistMax)
    normreg.registerVN(normParV, rp["useTPInvDistVec"][1], estat, "pTPInvDistVec", "wTPInvDistVec", "pTPInvDist", "wTPInvDist", defaultMean, defalutStdLen, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=17

if rp["useTPInvDist2Len"][0] or rp["useTPInvDist2Vec"][0]:
  upFeatures=(1.0/(0.001*rp["neighborCutoff"]+partDevDistLen)**2)
  uwFeatures=(1.0/(0.001*rp["neighborCutoff"]+devDistLen)**2)
  estat.trackE("pTPInvDist2", upFeatures, epochNr, lockStat)
  estat.trackE("wTPInvDist2", uwFeatures, epochNr, lockStat)
  
  if rp["useTPInvDist2Len"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(devDistMax)**2
    normreg.registerSN(normParS, rp["useTPInvDist2Len"][1], estat, "pTPInvDist2", "wTPInvDist2", defaultMean, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=18
  
  if rp["useTPInvDist2Vec"][0]:
    upFeatures=(partDevDistVec/(0.001*rp["neighborCutoff"]+partDevDistLen)**3)
    uwFeatures=(devDistVec/(0.001*rp["neighborCutoff"]+devDistLen)**3)
    estat.trackE("pTPInvDist2Vec", upFeatures, epochNr, lockStat)
    estat.trackE("wTPInvDist2Vec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(devDistMax)**2
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(devDistMax)**2
    normreg.registerVN(normParV, rp["useTPInvDist2Vec"][1], estat, "pTPInvDist2Vec", "wTPInvDist2Vec", "pTPInvDist2", "wTPInvDist2", defaultMean, defalutStdLen, defaultStd, len(particleNodeDataList), len(wallNodeDataList))
    particleNodeDataList.append(upFeatures)
    wallNodeDataList.append(uwFeatures)
    if dbg:
      dbgInt=19



#---



if rp["useNormalVec"][0]:
  particleNodeDataList.append(     torch.full((nrParticles,3), 0.0, device=device, dtype=torch.float32)     )
  wallNodeDataList.append(     (nvec*(rp["multNormV"]/2.0))[takeDist[:,1]]*particleWallWeight     )
  if dbg:
    dbgInt=20

if rp["useOneHotPE"][0]: #particle encoding
  particleNodeDataList.append(     torch.cat([torch.full((nrParticles,1), rp["multParticle"]*1.0, device=device, dtype=torch.float32), torch.full((nrParticles,1), rp["multWall"]*0.0, device=device, dtype=torch.float32)], 1)     )
  wallNodeDataList.append(     torch.cat([torch.full((nrWallParticles,1), rp["multParticle"]*0.0, device=device, dtype=torch.float32), torch.full((nrWallParticles,1), rp["multWall"]*1.0, device=device, dtype=torch.float32)*particleWallWeight], 1)     )
  if dbg:
    dbgInt=21



#---



if rp["useDistLen"][0] or rp["useDistVec"][0]:
  upFeatures=partDistLenPerturbated
  uwFeatures=torch.cat([wallDistLen], 0)
  estat.trackE("pDist", upFeatures, epochNr, lockStat)
  estat.trackE("wDist", uwFeatures, epochNr, lockStat)
  
  if rp["useDistLen"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/rp["neighborCutoff"])
    normreg.registerSE(normParS, rp["useDistLen"][1], estat, "pDist", "wDist", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=22
  
  if rp["useDistVec"][0]:
    upFeatures=partDistVecPerturbated
    uwFeatures=torch.cat([-wallDistVec], 0)
    estat.trackE("pDistVec", upFeatures, epochNr, lockStat)
    estat.trackE("wDistVec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(1.0/rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/rp["neighborCutoff"])
    normreg.registerVE(normParV, rp["useDistVec"][1], estat, "pDistVec", "wDistVec", "pDist", "wDist", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=23

if rp["useInvDistLen"][0] or rp["useInvDistVec"][0]:
  upFeatures=1.0/(0.001*rp["neighborCutoff"]+partDistLenPerturbated)
  uwFeatures=torch.cat([1.0/(0.001*rp["neighborCutoff"]+wallDistLen)], 0)
  estat.trackE("pInvDist", upFeatures, epochNr, lockStat)
  estat.trackE("wInvDist", uwFeatures, epochNr, lockStat)
  
  if rp["useInvDistLen"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])
    normreg.registerSE(normParS, rp["useInvDistLen"][1], estat, "pInvDist", "wInvDist", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=24
  
  if rp["useInvDistVec"][0]:
    upFeatures=(partDistVecPerturbated/(0.001*rp["neighborCutoff"]+partDistLenPerturbated)**2)
    uwFeatures=torch.cat([-(wallDistVec/(0.001*rp["neighborCutoff"]+wallDistLen)**2)], 0)
    estat.trackE("pInvDistVec", upFeatures, epochNr, lockStat)
    estat.trackE("wInvDistVec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])
    normreg.registerVE(normParV, rp["useInvDistVec"][1], estat, "pInvDistVec", "wInvDistVec", "pInvDist", "wInvDist", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=25

if rp["useInvDist2Len"][0] or rp["useInvDist2Vec"][0]:
  upFeatures=1.0/((0.001*rp["neighborCutoff"]+partDistLenPerturbated))**2
  uwFeatures=torch.cat([1.0/((0.001*rp["neighborCutoff"]+wallDistLen))**2], 0)
  estat.trackE("pInvDist2", upFeatures, epochNr, lockStat)
  estat.trackE("wInvDist2", uwFeatures, epochNr, lockStat)
  
  if rp["useInvDist2Len"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])**2
    normreg.registerSE(normParS, rp["useInvDist2Len"][1], estat, "pInvDist2", "wInvDist2", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=26
  
  if rp["useInvDist2Vec"][0]:
    upFeatures=(partDistVecPerturbated/(0.001*rp["neighborCutoff"]+partDistLenPerturbated)**3)
    uwFeatures=torch.cat([-wallDistVec/((0.001*rp["neighborCutoff"]+wallDistLen)**3)], 0)
    estat.trackE("pInvDist2Vec", upFeatures, epochNr, lockStat)
    estat.trackE("wInvDist2Vec", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(rp["neighborCutoff"])**2
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])**2
    normreg.registerVE(normParV, rp["useInvDist2Vec"][1], estat, "pInvDist2Vec", "wInvDist2Vec", "pInvDist2", "wInvDist2", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=27

if rp["useDistLenVMod"][0] or rp["useDistVecVMod"][0]:
  upFeatures=partDistLenVMod
  uwFeatures=torch.cat([wallDistLenVMod], 0)
  estat.trackE("pDistVMod", upFeatures, epochNr, lockStat)
  estat.trackE("wDistVMod", uwFeatures, epochNr, lockStat)
  
  if rp["useDistLenVMod"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/rp["neighborCutoff"])
    normreg.registerSE(normParS, rp["useDistLenVMod"][1], estat, "pDistVMod", "wDistVMod", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=28
  
  if rp["useDistVecVMod"][0]:
    upFeatures=partDistVecVMod
    uwFeatures=torch.cat([-wallDistVecVMod], 0)
    estat.trackE("pDistVecVMod", upFeatures, epochNr, lockStat)
    estat.trackE("wDistVecVMod", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(1.0/rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(1.0/rp["neighborCutoff"])
    normreg.registerVE(normParV, rp["useDistVecVMod"][1], estat, "pDistVecVMod", "wDistVecVMod", "pDistVMod", "wDistVMod", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=29

if rp["useInvDistLenVMod"][0] or rp["useInvDistVecVMod"][0]:
  upFeatures=1.0/(0.001*rp["neighborCutoff"]+partDistLenVMod)
  uwFeatures=torch.cat([1.0/(0.001*rp["neighborCutoff"]+wallDistLenVMod)], 0)
  estat.trackE("pInvDistVMod", upFeatures, epochNr, lockStat)
  estat.trackE("wInvDistVMod", uwFeatures, epochNr, lockStat)
  
  if rp["useInvDistLenVMod"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])
    normreg.registerSE(normParS, rp["useInvDistLenVMod"][1], estat, "pInvDistVMod", "wInvDistVMod", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=30
  
  if rp["useInvDistVecVMod"][0]:
    upFeatures=(partDistVecVMod/(0.001*rp["neighborCutoff"]+partDistLenVMod)**2)
    uwFeatures=torch.cat([-(wallDistVecVMod/(0.001*rp["neighborCutoff"]+wallDistLenVMod)**2)], 0)
    estat.trackE("pInvDistVecVMod", upFeatures, epochNr, lockStat)
    estat.trackE("wInvDistVecVMod", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(rp["neighborCutoff"])
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])
    normreg.registerVE(normParV, rp["useInvDistVecVMod"][1], estat, "pInvDistVecVMod", "wInvDistVecVMod", "pInvDistVMod", "wInvDistVMod", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=31

if rp["useInvDist2LenVMod"][0] or rp["useInvDist2VecVMod"][0]:
  upFeatures=1.0/(0.001*rp["neighborCutoff"]+partDistLenVMod)**2
  uwFeatures=torch.cat([1.0/((0.001*rp["neighborCutoff"]+wallDistLenVMod))**2], 0)
  estat.trackE("pInvDist2VMod", upFeatures, epochNr, lockStat)
  estat.trackE("wInvDist2VMod", uwFeatures, epochNr, lockStat)
  
  if rp["useInvDist2LenVMod"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])**2
    normreg.registerSE(normParS, rp["useInvDist2LenVMod"][1], estat, "pInvDist2VMod", "wInvDist2VMod", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=32
  
  if rp["useInvDist2VecVMod"][0]:
    upFeatures=(partDistVecVMod/(0.001*rp["neighborCutoff"]+partDistLenVMod)**3)
    uwFeatures=torch.cat([-(wallDistVecVMod/(0.001*rp["neighborCutoff"]+wallDistLenVMod)**3)], 0)
    estat.trackE("pInvDist2VecVMod", upFeatures, epochNr, lockStat)
    estat.trackE("wInvDist2VecVMod", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*(rp["neighborCutoff"])**2
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(rp["neighborCutoff"])**2
    normreg.registerVE(normParV, rp["useInvDist2VecVMod"][1], estat, "pInvDist2VecVMod", "wInvDist2VecVMod", "pInvDist2VMod", "wInvDist2VMod", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=33



#---



if rp["useUnitDistVec"][0]:
  upFeatures=partUnitDistVecPerturbated
  uwFeatures=torch.cat([-wallUnitDistVec], 0)
  estat.trackE("pUnitDistVec", upFeatures, epochNr, lockStat)
  estat.trackE("wUnitDistVec", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)
  normreg.registerSE(normParS, rp["useUnitDistVec"][1], estat, "pUnitDistVec", "wUnitDistVec", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
  particleEdgeDataList.append(upFeatures)
  wallEdgeDataList.append(uwFeatures)
  if dbg:
    dbgInt=34



#---



if rp["useProjectedUnitDistLenSenders"][0] or rp["useProjectedUnitDistVecSenders"][0]:
  upFeatures=projectedPartDistLenSendersPerturbated
  uwFeatures=torch.cat([projectedWallDistLenPerturbated*0.0], 0)
  estat.trackE("pProjectedUnitDistLenSenders", upFeatures, epochNr, lockStat)
  estat.trackE("wProjectedUnitDistLenSenders", uwFeatures, epochNr, lockStat)
  
  if rp["useProjectedUnitDistLenSenders"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*torch.from_numpy(stdVelLen).float().to(device)
    normreg.registerSE(normParS, rp["useProjectedUnitDistLenSenders"][1], estat, "pProjectedUnitDistLenSenders", "wProjectedUnitDistLenSenders", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=35
  
  if rp["useProjectedUnitDistVecSenders"][0]:
    upFeatures=projectedPartDistVecSendersPerturbated
    uwFeatures=torch.cat([-projectedWallDistVecPerturbated*0.0], 0)
    estat.trackE("pProjectedUnitDistVecSenders", upFeatures, epochNr, lockStat)
    estat.trackE("wProjectedUnitDistVecSenders", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*torch.from_numpy(stdVelLen).float().to(device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*torch.from_numpy(stdVelLen).float().to(device)
    normreg.registerVE(normParV, rp["useProjectedUnitDistVecSenders"][1], estat, "pProjectedUnitDistVecSenders", "wProjectedUnitDistVecSenders", "pProjectedUnitDistLenSenders", "wProjectedUnitDistLenSenders", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=36

if rp["useProjectedUnitDistLen2Senders"][0]:
  upFeatures=(projectedPartDistLenSendersPerturbated)**2
  uwFeatures=(torch.cat([projectedWallDistLenPerturbated*0.0], 0))**2
  estat.trackE("pProjectedUnitDist2Senders", upFeatures, epochNr, lockStat)
  estat.trackE("wProjectedUnitDist2Senders", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(torch.from_numpy(stdVelLen).float().to(device)**2)
  normreg.registerSE(normParS, rp["useProjectedUnitDistLen2Senders"][1], estat, "pProjectedUnitDist2Senders", "wProjectedUnitDist2Senders", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
  particleEdgeDataList.append(upFeatures)
  wallEdgeDataList.append(uwFeatures)
  if dbg:
    dbgInt=37

if rp["useProjectedUnitDistLenReceivers"][0] or rp["useProjectedUnitDistVecReceivers"][0]:
  upFeatures=projectedPartDistLenReceiversPerturbated
  uwFeatures=torch.cat([projectedWallDistLenPerturbated], 0)
  estat.trackE("pProjectedUnitDistLenReceivers", upFeatures, epochNr, lockStat)
  estat.trackE("wProjectedUnitDistLenReceivers", uwFeatures, epochNr, lockStat)
  
  if rp["useProjectedUnitDistLenReceivers"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*torch.from_numpy(stdVelLen).float().to(device)
    normreg.registerSE(normParS, rp["useProjectedUnitDistLenReceivers"][1], estat, "pProjectedUnitDistLenReceivers", "wProjectedUnitDistLenReceivers", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=38
  
  if rp["useProjectedUnitDistVecReceivers"][0]:
    upFeatures=projectedPartDistVecReceiversPerturbated
    uwFeatures=torch.cat([-projectedWallDistVecPerturbated], 0)
    estat.trackE("pProjectedUnitDistVecReceivers", upFeatures, epochNr, lockStat)
    estat.trackE("wProjectedUnitDistVecReceivers", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*torch.from_numpy(stdVelLen).float().to(device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*torch.from_numpy(stdVelLen).float().to(device)
    normreg.registerVE(normParV, rp["useProjectedUnitDistVecReceivers"][1], estat, "pProjectedUnitDistVecReceivers", "wProjectedUnitDistVecReceivers", "pProjectedUnitDistLenReceivers", "wProjectedUnitDistLenReceivers", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=39

if rp["useProjectedUnitDistLen2Receivers"][0]:
  upFeatures=(projectedPartDistLenReceiversPerturbated)**2
  uwFeatures=(torch.cat([projectedWallDistLenPerturbated], 0))**2
  estat.trackE("pProjectedUnitDist2Receivers", upFeatures, epochNr, lockStat)
  estat.trackE("wProjectedUnitDist2Receivers", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(torch.from_numpy(stdVelLen).float().to(device)**2)
  normreg.registerSE(normParS, rp["useProjectedUnitDistLen2Receivers"][1], estat, "pProjectedUnitDist2Receivers", "wProjectedUnitDist2Receivers", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
  particleEdgeDataList.append(upFeatures)
  wallEdgeDataList.append(uwFeatures)
  if dbg:
    dbgInt=40

if rp["useProjectedPartDistLenSum"][0] or rp["useProjectedPartDistVecSum"][0]:
  upFeatures=projectedPartDistLenSumPerturbated
  uwFeatures=torch.cat([projectedWallDistLenPerturbated], 0)
  estat.trackE("pProjectedPartDistLenSum", upFeatures, epochNr, lockStat)
  estat.trackE("wProjectedPartDistLenSum", uwFeatures, epochNr, lockStat)
  
  if rp["useProjectedPartDistLenSum"][0]:
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*torch.from_numpy(stdVelLen).float().to(device)
    normreg.registerSE(normParS, rp["useProjectedPartDistLenSum"][1], estat, "pProjectedPartDistLenSum", "wProjectedPartDistLenSum", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=41
  
  if rp["useProjectedPartDistVecSum"][0]:
    upFeatures=projectedPartDistVecSumPerturbated
    uwFeatures=torch.cat([-projectedWallDistVecPerturbated], 0)
    estat.trackE("pProjectedPartDistVecSum", upFeatures, epochNr, lockStat)
    estat.trackE("wProjectedPartDistVecSum", uwFeatures, epochNr, lockStat)
    
    defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
    defalutStdLen=torch.ones((1,)+upFeatures.shape[1:-1]+(1,), device=device)*torch.from_numpy(stdVelLen).float().to(device)
    defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*torch.from_numpy(stdVelLen).float().to(device)
    normreg.registerVE(normParV, rp["useProjectedPartDistVecSum"][1], estat, "pProjectedPartDistVecSum", "wProjectedPartDistVecSum", "pProjectedPartDistLenSum", "wProjectedPartDistLenSum", defaultMean, defalutStdLen, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
    particleEdgeDataList.append(upFeatures)
    wallEdgeDataList.append(uwFeatures)
    if dbg:
      dbgInt=42

if rp["useProjectedPartDistLen2Sum"][0]:
  upFeatures=(projectedPartDistLenSumPerturbated)**2
  uwFeatures=(torch.cat([projectedWallDistLenPerturbated], 0))**2
  estat.trackE("pProjectedPartDist2Sum", upFeatures, epochNr, lockStat)
  estat.trackE("wProjectedPartDist2Sum", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)*(torch.from_numpy(stdVelLen).float().to(device)**2)
  normreg.registerSE(normParS, rp["useProjectedPartDistLen2Sum"][1], estat, "pProjectedPartDist2Sum", "wProjectedPartDist2Sum", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
  particleEdgeDataList.append(upFeatures)
  wallEdgeDataList.append(uwFeatures)
  if dbg:
    dbgInt=43



#---



if rp["useAngle"][0]:
  upFeatures=torch.zeros((nrParticleDistances,1), device=device)
  uwFeatures=torch.cat([torch.reshape(angle, (-1,1))], 0)
  estat.trackE("pAngle", upFeatures, epochNr, lockStat)
  estat.trackE("wAngle", uwFeatures, epochNr, lockStat)
  
  defaultMean=torch.zeros((1,)+upFeatures.shape[1:], device=device)
  defaultStd=torch.ones((1,)+upFeatures.shape[1:], device=device)
  normreg.registerSE(normParS, rp["useAngle"][1], estat, "pAngle", "wAngle", defaultMean, defaultStd, len(particleEdgeDataList), len(wallEdgeDataList))
  particleEdgeDataList.append(upFeatures)
  wallEdgeDataList.append(uwFeatures)
  if dbg:
    dbgInt=44



#---



if epochNr==0:
  estat.endTrack()
normreg.normalize(particleNodeDataList, wallNodeDataList, particleEdgeDataList, wallEdgeDataList)