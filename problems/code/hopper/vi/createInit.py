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

import os
import pathlib
import shutil
import multiprocessing as mp
import numpy as np
import math
import pickle
import stl



gitRoot=os.path.join(os.environ['HOME'], "git", "bgnn")
simGitDir=os.path.join(gitRoot, "problems")

templateGitDir=os.path.join(simGitDir, "templates", "hopper", "vi")
stlGitDir=os.path.join(templateGitDir, "stl")

simDir=os.path.join("/system/user/mayr-data/BGNN/hopper/")
runDir=os.path.join(simDir, "runsi")




nrSimRunsStart=0
nrSimRuns=40
nrSimRunsStop=nrSimRunsStart+nrSimRuns



destPathBase=runDir



cutFnct=np.arange(10)
if os.path.exists(os.path.join(destPathBase, "cutFnctUse.pckl")):
  parUsageFile=open(os.path.join(destPathBase, "cutFnctUse.pckl"), "rb")
  cutFnctUse=pickle.load(parUsageFile)
  parUsageFile.close()
  if len(cutFnctUse)<len(cutFnct):
    cutFnctUse=np.concatenate([cutFnctUse, np.zeros(len(cutFnct)-len(cutFnctUse), dtype=np.int64)])
else:
  cutFnctUse=np.zeros(len(cutFnct), dtype=np.int64)



angleRange=np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40]) #40 seems to be the maximum
angleLower=angleRange[:-1]
angleUpper=angleRange[1:]

angleInd=np.arange(len(angleRange)-1)
if os.path.exists(os.path.join(destPathBase, "angleUse.pckl")):
  parUsageFile=open(os.path.join(destPathBase, "angleUse.pckl"), "rb")
  angleUse=pickle.load(parUsageFile)
  parUsageFile.close()
  if len(angleUse)<len(angleInd):
    angleUse=np.concatenate([angleUse, np.zeros(len(angleInd)-len(angleUse), dtype=np.int64)])
else:
  angleUse=np.zeros(len(angleInd), dtype=np.int64)



hyperpar=[cutFnct, angleInd]
hyperparUse=[cutFnctUse, angleUse]
hyperparInd=[]
for hInd in range(0, len(hyperpar)):
  hyperparInd.append(np.arange(len(hyperpar[hInd])))

for simInd in range(nrSimRunsStart, nrSimRunsStop):
  for hInd in range(0, len(hyperpar)):
    parOrder=np.random.permutation(len(hyperparUse[hInd]))
    parInd=hyperparInd[hInd][parOrder]
    parUse=hyperparUse[hInd][parOrder]
    parOrder=np.argsort(parUse)
    parInd=parInd[parOrder]
    parUse=parUse[parOrder]
    hyperparInd[hInd]=parInd
    parUse[0]=parUse[0]+1
    hyperparUse[hInd]=parUse
  
  
  
  kch=np.random.choice(2,1)[0]
  if kch==0:
    nokeep=np.random.triangular(0.1, 0.1, 0.5, 1)
  else:
    nokeep=1.0
  nokeep=np.random.triangular(0.0005, 0.1, 0.5, 1)
  
  
  
  velX=np.random.normal(0.5, 0.5)*np.sign(np.random.normal())
  velY=np.random.normal(0.5, 0.5)*np.sign(np.random.normal())
  zch=np.random.choice(2,1)[0]
  if zch==0:
    velZ=np.random.uniform(0.0, 1.0, 1)[0]
  elif zch==1:
    velZ=np.random.uniform(0.0, 0.0, 1)[0]
  velX=0.0
  velY=0.0
  velZ=0.0
  
  cutFnctInd=hyperparInd[0][0]
  
  
  angleInd=hyperparInd[1][0]; angle=np.random.uniform(angleLower[angleInd], angleUpper[angleInd])
  
  
  
  # Saved value - Hack for reproducibility - not originally included and can be set under comment
  destPath=os.path.join(destPathBase, str(simInd))
  parFile=open(os.path.join(destPath, "angle.pckl"),"rb"); angle=pickle.load(parFile); parFile.close()
  if angle>180:
    angle=angle-360
  #---
  
  
  
  a=0.06
  h=0.3
  
  add=h*math.sin(angle*((2.0*math.pi)/360.0))
  #addHeight=h*math.cos(angle*((2.0*math.pi)/360.0))-h
  
  if angle<0:
    angle=angle+360
  
  if add<0:
    scaleInletX=(a+2*add)/a
    borderX=a/2.0
  else:
    scaleInletX=1.0
    borderX=a/2.0+add
  
  timeScale=1.0/scaleInletX
  
  
  
  holeSize=np.random.uniform(0.012, 0.038)
  
  
  
  
  destPath=os.path.join(destPathBase, str(simInd))
  
  if not os.path.exists(destPath):
    os.mkdir(destPath)
    os.mkdir(os.path.join(destPath, "meshes"))
    os.mkdir(os.path.join(destPath, "post"))
    os.mkdir(os.path.join(destPath, "restart"))
  
  #parFile=open(os.path.join(destPath, "nokeepProb.pckl"),"wb"); pickle.dump(nokeep, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velX.pckl"),"wb"); pickle.dump(velX, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velY.pckl"),"wb"); pickle.dump(velY, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velZ.pckl"),"wb"); pickle.dump(velZ, parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "cutMainInd.pckl"),"wb"); pickle.dump(cutFnct[cutFnctInd], parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "angle.pckl"),"wb"); pickle.dump(angle, parFile); parFile.close()  
  #parFile=open(os.path.join(destPath, "holeSize.pckl"),"wb"); pickle.dump(holeSize, parFile); parFile.close()
  
  # Saved values - Hack for reproducibility - not originally included and can be set under comment
  parFile=open(os.path.join(destPath, "nokeepProb.pckl"),"rb"); nokeep=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cutMainInd.pckl"),"rb"); cutMainInd=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "holeSize.pckl"),"rb"); holeSize=pickle.load(parFile); parFile.close()
  #---
  
  outMeshDir=os.path.join(destPath, "meshes")
  if not os.path.exists(outMeshDir):
    os.makedirs(outMeshDir)
  
  inSTLPath=stlGitDir
  outSTLPath=outMeshDir
  
  scaleX=borderX*2.0*1.1
  scaleY=0.065*2.0*1.1
  scaleZ=((0.046+0.300)*1.1+(0.046+0.300)*(1.0/3.0))
  #moveX=-0.030
  moveX=-borderX*1.1
  moveY=-0.065*1.1
  moveZ=-(0.046+0.300)*(1.0/3.0)
  
  border=stl.mesh.Mesh.from_file(os.path.join(inSTLPath, 'border.stl'))
  border.vectors[:,:,0]=border.vectors[:,:,0]*scaleX+moveX
  border.vectors[:,:,1]=border.vectors[:,:,1]*scaleY+moveY
  border.vectors[:,:,2]=border.vectors[:,:,2]*scaleZ+moveZ
  border.save(os.path.join(outSTLPath, 'border.stl'))
  
  inlet=stl.mesh.Mesh.from_file(os.path.join(inSTLPath, 'inlet.stl'))
  inlet.vectors[:,:,0]=inlet.vectors[:,:,0]*scaleInletX
  inlet.save(os.path.join(outSTLPath, 'inlet.stl'))
  
  regionX0, regionY0, regionZ0=border.vectors.min(axis=(0,1))*1.01
  regionX1, regionY1, regionZ1=border.vectors.max(axis=(0,1))*1.01
  regionX0=min(regionX0, -0.16)
  regionX1=max(regionX1, 0.16)
  regionY0=min(regionY0, -0.07)
  regionY1=max(regionY1, 0.07)
  regionZ0=min(regionZ0, -0.1)
  regionZ1=max(regionZ1, 0.35)
  
  parFile=open(os.path.join(destPath, "region.pckl"),"wb")
  pickle.dump(regionX0, parFile)
  pickle.dump(regionX1, parFile)
  pickle.dump(regionY0, parFile)
  pickle.dump(regionY1, parFile)
  pickle.dump(regionZ0, parFile)
  pickle.dump(regionZ1, parFile)
  parFile.close()
  
  myTemp=pathlib.Path(os.path.join(templateGitDir, "parameters0.tmpl")).read_text()
  
  rt_init=1.0*timeScale
  holeSize=holeSize
  angle=angle
  
  ym1=5.e6
  ym2=5.e6
  pr1=0.4
  pr2=0.4
  
  cr11=0.95
  cr12=0.90
  cr21=0.90
  cr22=0.85
  cf11=0.45
  cf12=0.30
  cf21=0.30
  cf22=0.20
  crf11=0.020
  crf12=0.015
  crf21=0.015
  crf22=0.010
  
  nrParticles=20000
  ps=2e-3
  density=2500
  
  #writeFile=pathlib.Path(os.path.join(destPath, "parameters.par"))
  #writeFile.write_text(myTemp.format(regionX0=regionX0, regionY0=regionY0, regionZ0=regionZ0, regionX1=regionX1, regionY1=regionY1, regionZ1=regionZ1, rt_init=rt_init, holeSize=holeSize, angle=angle, ym1=ym1, ym2=ym2, pr1=pr1, pr2=pr2, cr11=cr11, cr12=cr12, cr21=cr21, cr22=cr22, cf11=cf11, cf12=cf12, cf21=cf21, cf22=cf22, crf11=crf11, crf12=crf12, crf21=crf21, crf22=crf22, nrParticles=nrParticles, ps=ps, density=density))
  
  
  
  shutil.copy(os.path.join(templateGitDir, "initScript"), os.path.join(destPath, "initScript"))
  shutil.copy(os.path.join(templateGitDir, "mainScript"), os.path.join(destPath, "mainScript"))
  shutil.copy(os.path.join(inSTLPath, 'hopper_bottom.stl'), os.path.join(outSTLPath, 'hopper_bottom.stl'))
  shutil.copy(os.path.join(inSTLPath, 'hopper_plate.stl'), os.path.join(outSTLPath, 'hopper_plate.stl'))
  shutil.copy(os.path.join(inSTLPath, 'hopper_side_wall.stl'), os.path.join(outSTLPath, 'hopper_side_wall.stl'))
  shutil.copy(os.path.join(inSTLPath, 'hopper_slanted_wall.stl'), os.path.join(outSTLPath, 'hopper_slanted_wall.stl'))

for hInd in range(0, len(hyperpar)):
  parInd=hyperparInd[hInd]
  parUse=hyperparUse[hInd]
  parOrder=np.argsort(parInd)
  parInd=parInd[parOrder]
  parUse=parUse[parOrder]
  hyperparInd[hInd]=parInd
  hyperparUse[hInd]=parUse

cutFnctUse=hyperparUse[0]
parUsageFile=open(os.path.join(destPathBase, "cutFnctUse.pckl"),"wb")
pickle.dump(cutFnctUse, parUsageFile)
parUsageFile.close()

angleUse=hyperparUse[1]
parUsageFile=open(os.path.join(destPathBase, "angleUse.pckl"),"wb")
pickle.dump(angleUse, parUsageFile)
parUsageFile.close()
