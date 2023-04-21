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
import primesieve
import more_itertools
import copy



gitRoot=os.path.join(os.environ['HOME'], "git", "bgnn")
simGitDir=os.path.join(gitRoot, "problems")

templateGitDir=os.path.join(simGitDir, "templates", "hopper", "v34")
stlGitDir=os.path.join(templateGitDir, "stl")

simDir=os.path.join("/system/user/mayr-data/BGNN/hopper/")
runDir=os.path.join(simDir, "runs4")




nrSimRunsStart=40
nrSimRuns=15
nrSimRunsStop=nrSimRunsStart+nrSimRuns




destPathBase=runDir



def initFunc(nrValues, destPathBase, name):
  ind=np.arange(nrValues)
  if os.path.exists(os.path.join(destPathBase, name+"Use.pckl")):
    parUsageFile=open(os.path.join(destPathBase, name+"Use.pckl"), "rb")
    use=pickle.load(parUsageFile)
    parUsageFile.close()
    if len(use)<len(ind):
      use=np.concatenate([use, np.zeros(len(ind)-len(use), dtype=np.int64)])
  else:
    use=np.zeros(len(ind), dtype=np.int64)
  return ind, use, name

def finFunc(destPathBase, name):
  use=hyperparUse[hyperparName[name]]
  parUsageFile=open(os.path.join(destPathBase, name+"Use.pckl"),"wb")
  pickle.dump(use, parUsageFile)
  parUsageFile.close()


openInitPar=[0]
openMainPar=[0]
#angleLower=[ 0,  0,  0,  4,  4,  4,  8,  8,  8, 12, 12, 12, 16, 16]
#angleUpper=[ 4,  4,  4,  8,  8,  8, 12, 12, 12, 16, 16, 16, 20, 20]
angleLower=[20, 20, 20, 20, 22, 22, 22, 24, 24, 24, 24, 24, 28, 28]
angleUpper=[22, 22, 22, 22, 24, 24, 24, 28, 28, 28, 28, 28, 32, 32]
holeSizeLower=[0.4/1.5] #..., starting from 0.4, such that numbers between bottomResizeLower and bottomResizeUpper get valid ==> >=min
holeSizeUpper=[1.0]
bottomResizeLower=[0.4] #devided by holeSize
bottomResizeUpper=[0.8] #devided by holeSize
ymInitPar=[[5.e6, 5.e6, 5.e6]]
ymMainPar=[[5.e6, 5.e6, 5.e6]]
prInitPar=[[0.4, 0.4, 0.4]]
prMainPar=[[0.4, 0.4, 0.4]]
crInitPar=[[[0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]]]
crMainPar=[[[0.20, 0.20, 0.20], 
            [0.20, 0.20, 0.20], 
            [0.20, 0.20, 0.20]]]
#crMainPar=copy.deepcopy(crInitPar)
cfInitPar=[[[0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]]]
cfMainPar=[[[0.05, 0.05, 0.05],
            [0.05, 0.01, 0.01],
            [0.05, 0.01, 0.01]]]
#cfMainPar=copy.deepcopy(crInitPar)
crfInitPar=[[[0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]]]
crfMainPar=[[[0.05, 0.05, 0.05],
            [0.05, 0.01, 0.01],
            [0.05, 0.01, 0.01]]]
#crfMainPar=copy.deepcopy(crfInitPar)
cedInitPar=[[[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]]
cedMainPar=[[[0.00000,   10**5*0.0, 10**5*0.0],
             [10**5*0.0, 10**5*0.0, 10**5*0.0],
             [10**5*0.0, 10**5*0.0, 10**5*0.0]]]
#cedMainPar=copy.deepcopy(cedInitPar)
densityInitPar=[
[None, [2500, 2500], [2500, 2500]]
]
densityMainPar=copy.deepcopy(densityInitPar)
radInitPar=[[None, [[2e-3, 2e-3], [2e-3, 2e-3]], [[2e-3, 2e-3], [2e-3, 2e-3]]]]
radMainPar=[[None, [[2e-3, 2e-3], [2e-3, 2e-3]], [[2e-3, 2e-3], [2e-3, 2e-3]]]]



hinit=[]
hinit.append(initFunc(10, destPathBase, "cut"))
hinit.append(initFunc(len(openInitPar), destPathBase, "openInit"))
hinit.append(initFunc(len(openMainPar), destPathBase, "openMain"))
hinit.append(initFunc(len(angleLower), destPathBase, "angle"))
hinit.append(initFunc(len(holeSizeLower), destPathBase, "holeSize"))
hinit.append(initFunc(len(bottomResizeLower), destPathBase, "bottomResize"))
hinit.append(initFunc(len(ymInitPar), destPathBase, "ymInit"))
hinit.append(initFunc(len(ymMainPar), destPathBase, "ymMain"))
hinit.append(initFunc(len(prInitPar), destPathBase, "prInit"))
hinit.append(initFunc(len(prMainPar), destPathBase, "prMain"))
hinit.append(initFunc(len(crInitPar), destPathBase, "crInit"))
hinit.append(initFunc(len(crMainPar), destPathBase, "crMain"))
hinit.append(initFunc(len(cfInitPar), destPathBase, "cfInit"))
hinit.append(initFunc(len(cfMainPar), destPathBase, "cfMain"))
hinit.append(initFunc(len(crfInitPar), destPathBase, "crfInit"))
hinit.append(initFunc(len(crfMainPar), destPathBase, "crfMain"))
hinit.append(initFunc(len(cedInitPar), destPathBase, "cedInit"))
hinit.append(initFunc(len(cedMainPar), destPathBase, "cedMain"))
hinit.append(initFunc(len(densityInitPar), destPathBase, "densityInit"))
hinit.append(initFunc(len(densityMainPar), destPathBase, "densityMain"))
hinit.append(initFunc(len(radInitPar), destPathBase, "radInit"))
hinit.append(initFunc(len(radMainPar), destPathBase, "radMain"))

hyperpar=[x[0] for x in hinit]
hyperparUse=[x[1] for x in hinit]
hyperparName={y:i for i,y in enumerate([x[2] for x in hinit])}
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
    nokeep=np.random.triangular(0.1, 0.1, 0.5, 1)[0]
  else:
    nokeep=1.0
  nokeep=np.random.triangular(0.0005, 0.1, 0.5, 1)[0]
  nokeep=np.random.triangular(0.0005, 0.3, 0.5, 1)[0]
  
  
  
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
  
  cutMainInd=hyperparInd[hyperparName["cut"]][0]
  
  openInitInd=hyperparInd[hyperparName["openInit"]][0]; openInit=openInitPar[openInitInd]
  openMainInd=hyperparInd[hyperparName["openMain"]][0]; openMain=openMainPar[openMainInd]
  
  angleInd=hyperparInd[hyperparName["angle"]][0]; angle=np.random.uniform(angleLower[angleInd], angleUpper[angleInd])
  
  
  
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
  
  
  
  holeSizeInd=hyperparInd[hyperparName["holeSize"]][0]; holeSize=np.random.uniform(holeSizeLower[holeSizeInd], holeSizeUpper[holeSizeInd])
  bottomResizeInd=hyperparInd[hyperparName["bottomResize"]][0]; bottomResize=np.random.uniform(bottomResizeLower[bottomResizeInd]/holeSize, min(1.5, bottomResizeUpper[bottomResizeInd]/holeSize))
  ymInitInd=hyperparInd[hyperparName["ymInit"]][0]; ymInit=ymInitPar[ymInitInd]
  ymMainInd=hyperparInd[hyperparName["ymMain"]][0]; ymMain=ymMainPar[ymMainInd]
  prInitInd=hyperparInd[hyperparName["prInit"]][0]; prInit=prInitPar[prInitInd]
  prMainInd=hyperparInd[hyperparName["prMain"]][0]; prMain=prMainPar[prMainInd]
  crInitInd=hyperparInd[hyperparName["crInit"]][0]; crInit=crInitPar[crInitInd]
  crMainInd=hyperparInd[hyperparName["crMain"]][0]; crMain=crMainPar[crMainInd]
  cfInitInd=hyperparInd[hyperparName["cfInit"]][0]; cfInit=cfInitPar[cfInitInd]
  cfMainInd=hyperparInd[hyperparName["cfMain"]][0]; cfMain=cfMainPar[cfMainInd]
  crfInitInd=hyperparInd[hyperparName["crfInit"]][0]; crfInit=crfInitPar[crfInitInd]
  crfMainInd=hyperparInd[hyperparName["crfMain"]][0]; crfMain=crfMainPar[crfMainInd]
  cedInitInd=hyperparInd[hyperparName["cedInit"]][0]; cedInit=cedInitPar[cedInitInd]
  cedMainInd=hyperparInd[hyperparName["cedMain"]][0]; cedMain=cedMainPar[cedMainInd]
  densityInitInd=hyperparInd[hyperparName["densityInit"]][0]; densityInit=densityInitPar[densityInitInd]
  densityMainInd=hyperparInd[hyperparName["densityMain"]][0]; densityMain=densityMainPar[densityMainInd]
  radInitInd=hyperparInd[hyperparName["radInit"]][0]; radInit=radInitPar[radInitInd]
  radMainInd=hyperparInd[hyperparName["radMain"]][0]; radMain=radMainPar[radMainInd]
  
  translationTable=[]
  for mytype in range(1,3):
    for mydens in range(0,2):
      for myrad in range(0,2):
        type0=mytype+1
        dmap0=densityInitPar[densityInitInd][mytype][mydens]
        rmap0=radInitPar[radInitInd][mytype][mydens][myrad]
        
        type1=mytype+1
        dmap1=densityMainPar[densityMainInd][mytype][mydens]
        rmap1=radMainPar[radMainInd][mytype][mydens][myrad]
        
        translationTable.append((type0, dmap0, rmap0, type1, dmap1, rmap1))
  
  destPath=os.path.join(destPathBase, str(simInd))
  
  if not os.path.exists(destPath):
    os.mkdir(destPath)
    os.mkdir(os.path.join(destPath, "meshes"))
    os.mkdir(os.path.join(destPath, "post"))
    os.mkdir(os.path.join(destPath, "restart"))
  
  #parFile=open(os.path.join(destPath, "nokeepProb.pckl"),"wb"); pickle.dump(nokeep, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velX.pckl"),"wb"); pickle.dump(velX, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velY.pckl"),"wb");pickle.dump(velY, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "velZ.pckl"),"wb"); pickle.dump(velZ, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "openInit.pckl"),"wb"); pickle.dump(openInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "openMain.pckl"),"wb"); pickle.dump(openMain, parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "cutMainInd.pckl"),"wb"); pickle.dump(cutMainInd, parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "angle.pckl"),"wb"); pickle.dump(angle, parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "holeSize.pckl"),"wb"); pickle.dump(holeSize, parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "bottomResize.pckl"),"wb"); pickle.dump(bottomResize, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "ymInit.pckl"),"wb"); pickle.dump(ymInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "ymMain.pckl"),"wb"); pickle.dump(ymMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "prInit.pckl"),"wb"); pickle.dump(prInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "prMain.pckl"),"wb"); pickle.dump(prMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crInit.pckl"),"wb"); pickle.dump(crInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crMain.pckl"),"wb"); pickle.dump(crMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cfInit.pckl"),"wb"); pickle.dump(cfInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cfMain.pckl"),"wb"); pickle.dump(cfMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crfInit.pckl"),"wb"); pickle.dump(crfInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "crfMain.pckl"),"wb"); pickle.dump(crfMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cedInit.pckl"),"wb"); pickle.dump(cedInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cedMain.pckl"),"wb"); pickle.dump(cedMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "densityInit.pckl"),"wb"); pickle.dump(densityInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "densityMain.pckl"),"wb"); pickle.dump(densityMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "radInit.pckl"),"wb"); pickle.dump(radInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "radMain.pckl"),"wb"); pickle.dump(radMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "translationTable.pckl"),"wb"); pickle.dump(translationTable, parFile); parFile.close()
  
  # Saved values - Hack for reproducibility - not originally included and can be set under comment
  parFile=open(os.path.join(destPath, "nokeepProb.pckl"),"rb"); nokeep=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "cutMainInd.pckl"),"rb"); cutMainInd=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "holeSize.pckl"),"rb"); holeSize=pickle.load(parFile); parFile.close()
  parFile=open(os.path.join(destPath, "bottomResize.pckl"),"rb"); bottomResize=pickle.load(parFile); parFile.close()
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
  border.save(os.path.join(outSTLPath, 'border.stl'), mode=stl.Mode.ASCII)
  
  inlet=stl.mesh.Mesh.from_file(os.path.join(inSTLPath, 'inlet.stl'))
  inlet.vectors[:,:,0]=inlet.vectors[:,:,0]*scaleInletX
  inlet.save(os.path.join(outSTLPath, 'inlet.stl'), mode=stl.Mode.ASCII)
  
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
  
  
  
  computeTimestep=2.5e-5
  statusTimestep=1e-3
  dumpTimeStep=1e-3
  runTimeInit=1.0*timeScale
  runTimeMain=1*3
  
  #myTemp=pathlib.Path(os.path.join(templateGitDir, "parameters0.tmpl")).read_text()
  
  
  
  primepool=np.random.permutation(primesieve.n_primes(10000, 10000))
  
  tempDictInit={
    "regionX0": regionX0,
    "regionX1": regionX1,
    "regionY0": regionY0,
    "regionY1": regionY1,
    "regionZ0": regionZ0,
    "regionZ1": regionZ1,
    "open": openInit,
    "dt": computeTimestep,
    "run_time": runTimeInit,
    "status_output": statusTimestep,
    "nrParticles": 20000,
    "angle": angle,
    "holeSize": 1.0, #holeSize,
    "bottomResize": bottomResize,
    "ym": ymInit,
    "pr": prInit,
    "cr": crInit,
    "cf": cfInit,
    "crf": crfInit,
    "ced": cedInit, 
    "density": densityInit,
    "rad": radInit,
    "tp": (np.ones((1,8))*0.12).tolist(), #np.identity(8).tolist(), #(np.ones((8,8))*0.12).tolist(),
    "templateSeed": primepool[0:8].tolist(),
    "pdistSeed": primepool[8:9].tolist(),
    "insSeed": primepool[9:10].tolist()
  }
  tempDictInit.update({"maxps": max([x for x in more_itertools.collapse(radInit) if x is not None])})
  
  myTempInit=\
  ["variable regionX0 equal {regionX0}"]+\
  ["variable regionX1 equal {regionX1}"]+\
  ["variable regionY0 equal {regionY0}"]+\
  ["variable regionY1 equal {regionY1}"]+\
  ["variable regionZ0 equal {regionZ0}"]+\
  ["variable regionZ1 equal {regionZ1}"]+\
  ["variable open equal {open}"]+\
  ["variable dt equal {dt}"]+\
  ["variable run_time equal {run_time}"]+\
  ["variable status_output equal {status_output}"]+\
  ["variable particles_to_insert equal {nrParticles}"]+\
  ["variable wall_angle equal {angle}"]+\
  ["variable hole_size equal {holeSize}"]+\
  ["variable particle_size equal {maxps}"]+\
  [f"variable ym{i+1} equal {{ym[{i}]}}" for i in range(len(tempDictInit["ym"]))]+\
  [f"variable pr{i+1} equal {{pr[{i}]}}" for i in range(len(tempDictInit["pr"]))]+\
  [f"variable cr{i+1}{j+1} equal {{cr[{i}][{j}]}}" for i in range(len(tempDictInit["cr"])) for j in range(len(tempDictInit["cr"][i]))]+\
  [f"variable cf{i+1}{j+1} equal {{cf[{i}][{j}]}}" for i in range(len(tempDictInit["cf"])) for j in range(len(tempDictInit["cf"][i]))]+\
  [f"variable crf{i+1}{j+1} equal {{crf[{i}][{j}]}}" for i in range(len(tempDictInit["crf"])) for j in range(len(tempDictInit["crf"][i]))]+\
  [f"variable ced{i+1}{j+1} equal {{ced[{i}][{j}]}}" for i in range(len(tempDictInit["ced"])) for j in range(len(tempDictInit["ced"][i]))]+\
  [f"variable density{i+1}{j+1} equal {{density[{i}][{j}]}}" for i in range(len(tempDictInit["density"]))  if tempDictInit["density"][i] is not None for j in range(len(tempDictInit["density"][i]))]+\
  [f"variable rad{i+1}{j+1}{k+1} equal {{rad[{i}][{j}][{k}]}}" for i in range(len(tempDictInit["rad"]))  if tempDictInit["rad"][i] is not None for j in range(len(tempDictInit["rad"][i])) for k in range(len(tempDictInit["rad"][i][j]))]+\
  [f"variable tp{i+1}{j+1} equal {{tp[{i}][{j}]}}" for i in range(len(tempDictInit["tp"])) for j in range(len(tempDictInit["tp"][i]))]+\
  [f"variable templateSeed{i+1} equal {{templateSeed[{i}]}}" for i in range(len(tempDictInit["templateSeed"]))]+\
  [f"variable pdistSeed{i+1} equal {{pdistSeed[{i}]}}" for i in range(len(tempDictInit["pdistSeed"]))]+\
  [f"variable insSeed{i+1} equal {{insSeed[{i}]}}" for i in range(len(tempDictInit["insSeed"]))]
  
  myTempInit='\n'.join(myTempInit)
  #writeFileInit=pathlib.Path(os.path.join(destPath, "parametersInit.par"))
  #writeFileInit.write_text(myTempInit.format(**tempDictInit))
  
  
  
  tempDictMain={
    "open": openMain,
    "dt": computeTimestep,
    "run_time": runTimeMain,
    "status_output": statusTimestep,
    "data_output": dumpTimeStep,
    "angle": angle,
    "holeSize": 1.0, #holeSize,
    "bottomResize": bottomResize,
    "ym": ymMain,
    "pr": prMain,
    "cr": crMain,
    "cf": cfMain,
    "crf": crfMain,
    "ced": cedMain,
    "density": densityMain,
    "rad": radMain
  }
  tempDictMain.update({"maxps": max([x for x in more_itertools.collapse(radMain) if x is not None])})
  
  myTempMain=\
  ["variable open equal {open}"]+\
  ["variable dt equal {dt}"]+\
  ["variable run_time equal {run_time}"]+\
  ["variable status_output equal {status_output}"]+\
  ["variable data_output equal {data_output}"]+\
  ["variable wall_angle equal {angle}"]+\
  ["variable hole_size equal {holeSize}"]+\
  ["variable particle_size equal {maxps}"]+\
  [f"variable ym{i+1} equal {{ym[{i}]}}" for i in range(len(tempDictMain["ym"]))]+\
  [f"variable pr{i+1} equal {{pr[{i}]}}" for i in range(len(tempDictMain["pr"]))]+\
  [f"variable cr{i+1}{j+1} equal {{cr[{i}][{j}]}}" for i in range(len(tempDictMain["cr"])) for j in range(len(tempDictMain["cr"][i]))]+\
  [f"variable cf{i+1}{j+1} equal {{cf[{i}][{j}]}}" for i in range(len(tempDictMain["cf"])) for j in range(len(tempDictMain["cf"][i]))]+\
  [f"variable crf{i+1}{j+1} equal {{crf[{i}][{j}]}}" for i in range(len(tempDictMain["crf"])) for j in range(len(tempDictMain["crf"][i]))]+\
  [f"variable ced{i+1}{j+1} equal {{ced[{i}][{j}]}}" for i in range(len(tempDictMain["ced"])) for j in range(len(tempDictMain["ced"][i]))]
  
  myTempMain='\n'.join(myTempMain)
  #writeFileMain=pathlib.Path(os.path.join(destPath, "parametersMain.par"))
  #writeFileMain.write_text(myTempMain.format(**tempDictMain))
  
  
  
  shutil.copy(os.path.join(templateGitDir, "initScript"), os.path.join(destPath, "initScript"))
  shutil.copy(os.path.join(templateGitDir, "mainScript"), os.path.join(destPath, "mainScript"))
  #shutil.copy(os.path.join(inSTLPath, 'hopper_bottom.stl'), os.path.join(outSTLPath, 'hopper_bottom.stl'))
  bottom=stl.mesh.Mesh.from_file(os.path.join(inSTLPath, 'hopper_bottom.stl'))
  unitVec=bottom.vectors/np.sqrt((bottom.vectors[:,:,0]**2+bottom.vectors[:,:,1]**2))[..., np.newaxis]
  oldRadiusPoint=19*unitVec
  newRadiusPoint=oldRadiusPoint*bottomResize
  diffVec=bottom.vectors-oldRadiusPoint
  bottom.vectors[:,:,0]=np.where(newRadiusPoint[:,:,0]>0.0, newRadiusPoint[:,:,0]+((30.-newRadiusPoint[:,:,0])/(30.-oldRadiusPoint[:,:,0]))*(diffVec[:,:,0]), newRadiusPoint[:,:,0]+((-30.-newRadiusPoint[:,:,0])/(-30.-oldRadiusPoint[:,:,0]))*(diffVec[:,:,0]))
  bottom.vectors[:,:,1]=np.where(newRadiusPoint[:,:,1]>0.0, newRadiusPoint[:,:,1]+((65.-newRadiusPoint[:,:,1])/(65.-oldRadiusPoint[:,:,1]))*(diffVec[:,:,1]), newRadiusPoint[:,:,1]+((-65.-newRadiusPoint[:,:,1])/(-65.-oldRadiusPoint[:,:,1]))*(diffVec[:,:,1]))
  bottom.save(os.path.join(outSTLPath, 'hopper_bottom.stl'), mode=stl.Mode.ASCII)
  #shutil.copy(os.path.join(inSTLPath, 'hopper_plate.stl'), os.path.join(outSTLPath, 'hopper_plate.stl'))
  plate=stl.mesh.Mesh.from_file(os.path.join(inSTLPath, 'hopper_plate.stl'))
  ind1=np.all(plate.vectors.reshape(-1,3)==np.array([-60, -65,0]),1)
  ind2=np.all(plate.vectors.reshape(-1,3)==np.array([-60, 65,0]),1)
  ind3=np.all(plate.vectors.reshape(-1,3)==np.array([60, -65,0]),1)
  ind4=np.all(plate.vectors.reshape(-1,3)==np.array([60, 65,0]),1)
  assert(ind1.sum()==6)
  assert(ind2.sum()==6)
  assert(ind3.sum()==6)
  assert(ind4.sum()==6)
  plateCopy=plate.vectors.reshape(-1,3).copy()
  plateCopy[ind1]=np.array([-30.0, -65.0, 0.0])
  plateCopy[ind2]=np.array([-30.0, 65.0, 0.0])
  plateCopy[ind3]=np.array([30.0, -65.0, 0.0])
  plateCopy[ind4]=np.array([30.0, 65.0, 0.0])
  plateCopy=plateCopy.reshape(plate.vectors.shape)
  unitVec=plateCopy/np.sqrt((plateCopy[:,:,0]**2+plateCopy[:,:,1]**2))[..., np.newaxis]
  oldRadiusPoint=19*unitVec
  newRadiusPoint=oldRadiusPoint*bottomResize
  diffVec=plateCopy-oldRadiusPoint
  plateCopy[:,:,0]=np.where(newRadiusPoint[:,:,0]>0.0, newRadiusPoint[:,:,0]+((30.-newRadiusPoint[:,:,0])/(30.-oldRadiusPoint[:,:,0]))*(diffVec[:,:,0]), newRadiusPoint[:,:,0]+((-30.-newRadiusPoint[:,:,0])/(-30.-oldRadiusPoint[:,:,0]))*(diffVec[:,:,0]))
  plateCopy[:,:,1]=np.where(newRadiusPoint[:,:,1]>0.0, newRadiusPoint[:,:,1]+((65.-newRadiusPoint[:,:,1])/(65.-oldRadiusPoint[:,:,1]))*(diffVec[:,:,1]), newRadiusPoint[:,:,1]+((-65.-newRadiusPoint[:,:,1])/(-65.-oldRadiusPoint[:,:,1]))*(diffVec[:,:,1]))
  plateCopy=plateCopy.reshape(-1,3)
  plateCopy[ind1]=plateCopy[ind1]*(1.0/holeSize)
  plateCopy[ind2]=plateCopy[ind2]*(1.0/holeSize)
  plateCopy[ind3]=plateCopy[ind3]*(1.0/holeSize)
  plateCopy[ind4]=plateCopy[ind4]*(1.0/holeSize)
  plateCopy=plateCopy.reshape(plate.vectors.shape)
  plate.vectors[:]=plateCopy*holeSize
  plate.save(os.path.join(outSTLPath, 'hopper_plate.stl'), mode=stl.Mode.ASCII)
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

for mykey in hyperparName.keys():
  finFunc(destPathBase, mykey)
