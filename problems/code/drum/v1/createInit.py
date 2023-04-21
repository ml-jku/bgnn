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

templateGitDir=os.path.join(simGitDir, "templates", "drum", "v12")
stlGitDir=os.path.join(templateGitDir, "stl")

simDir=os.path.join("/system/user/mayr-data/BGNN/drum/")
runDir=os.path.join(simDir, "runs1")




nrSimRunsStart=0
nrSimRuns=40
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
angleInitLower=[0]
angleInitUpper=[360]
angleMainLower=[0]
angleMainUpper=[0]
rpMainLower=[4]
rpMainUpper=[4]
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
cfMainPar=[[[0.95, 0.95, 0.95],
            [0.95, 0.45, 0.45],
            [0.95, 0.45, 0.45]]]
#cfMainPar=copy.deepcopy(crInitPar)
crfInitPar=[[[0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]]]
crfMainPar=[[[0.95, 0.95, 0.95],
            [0.95, 0.45, 0.45],
            [0.95, 0.45, 0.45]]]
#crfMainPar=copy.deepcopy(crfInitPar)
cedInitPar=[[[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]]
cedMainPar=[[[0.00000,   10**5*0.0, 10**5*0.0],
             [10**5*0.0, 10**5*1.0, 10**5*1.0],
             [10**5*0.0, 10**5*1.0, 10**5*1.0]]]
#cedMainPar=copy.deepcopy(cedInitPar)
densityInitPar=[[None, [2500, 2500], [2500, 2500]]]
densityMainPar=copy.deepcopy(densityInitPar)
radInitPar=[[None, [[1e-2, 1e-2], [1e-2, 1e-2]], [[1e-2, 1e-2], [1e-2, 1e-2]]]]
radMainPar=copy.deepcopy(radInitPar)
insRateInitPar=[1000]
insRateMainPar=[1000]



hinit=[]
hinit.append(initFunc(10, destPathBase, "cut"))
hinit.append(initFunc(len(openInitPar), destPathBase, "openInit"))
hinit.append(initFunc(len(openMainPar), destPathBase, "openMain"))
hinit.append(initFunc(len(angleInitLower), destPathBase, "angleInit"))
hinit.append(initFunc(len(angleMainLower), destPathBase, "angleMain"))
hinit.append(initFunc(len(rpMainLower), destPathBase, "rpMain"))
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
hinit.append(initFunc(len(insRateInitPar), destPathBase, "insRateInit"))
hinit.append(initFunc(len(insRateMainPar), destPathBase, "insRateMain"))

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
  
  velX=0.0
  velY=0.0
  velZ=0.0
  
  cutMainInd=hyperparInd[hyperparName["cut"]][0]
  
  angleInitInd=hyperparInd[hyperparName["angleInit"]][0]
  angleInit=np.random.uniform(angleInitLower[angleInitInd], angleInitUpper[angleInitInd])
  if angleInit<0:
    angleInit=angleInit+360
  
  angleMainInd=hyperparInd[hyperparName["angleMain"]][0]
  angleMain=np.random.uniform(angleMainLower[angleMainInd], angleMainUpper[angleMainInd])
  if angleMain<0:
    angleMain=angleMain+360
  
  rpMainInd=hyperparInd[hyperparName["rpMain"]][0]; rpMain=np.random.uniform(rpMainLower[rpMainInd], rpMainUpper[rpMainInd])
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
  insRateInitInd=hyperparInd[hyperparName["insRateInit"]][0]; insRateInit=insRateInitPar[insRateInitInd]
  
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
  #parFile=open(os.path.join(destPath, "cutMainInd.pckl"),"wb"); pickle.dump(cutMainInd, parFile); parFile.close()
  #parFile=open(os.path.join(destPath, "angleInit.pckl"),"wb"); pickle.dump(angleInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "angleMain.pckl"),"wb"); pickle.dump(angleMain, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "rpMain.pckl"),"wb"); pickle.dump(rpMain, parFile); parFile.close()
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
  parFile=open(os.path.join(destPath, "insRateInit.pckl"),"wb"); pickle.dump(insRateInit, parFile); parFile.close()
  parFile=open(os.path.join(destPath, "translationTable.pckl"),"wb"); pickle.dump(translationTable, parFile); parFile.close()
  
  
  
  computeTimestep=1.0e-5
  statusTimestep=1e-3
  dumpTimeStep=1e-3
  runTimeInit=5
  runTimeMain=10
  
  #myTemp=pathlib.Path(os.path.join(templateGitDir, "parameters0.tmpl")).read_text()
  
  
  
  primepool=np.random.permutation(primesieve.n_primes(10000, 10000))
  
  tempDictInit={
    "dt": computeTimestep,
    "run_time": runTimeInit,
    "status_output": statusTimestep,
    "nrParticles1": 2500,
    "nrParticles2": 2500,
    "angle": angleInit,
    "ym": ymInit,
    "pr": prInit,
    "cr": crInit,
    "cf": cfInit,
    "crf": crfInit,
    "ced": cedInit, 
    "density": densityInit,
    "rad": radInit,
    "tp": (np.array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]])).tolist(), #np.identity(8).tolist(), #(np.ones((8,8))*0.12).tolist(),
    "insRate": insRateInit,
    "templateSeed": primepool[0:8].tolist(),
    "pdistSeed": primepool[8:10].tolist(),
    "insSeed": primepool[10:12].tolist()
  }
  tempDictInit.update({"maxps": max([x for x in more_itertools.collapse(radInit) if x is not None])})
  
  myTempInit=\
  ["variable dt equal {dt}"]+\
  ["variable run_time equal {run_time}"]+\
  ["variable status_output equal {status_output}"]+\
  ["variable particles_to_insert1 equal {nrParticles1}"]+\
  ["variable particles_to_insert2 equal {nrParticles2}"]+\
  ["variable inclination_angle equal {angle}"]+\
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
  ["variable particle_insertion_rate equal {insRate}"]+\
  [f"variable templateSeed{i+1} equal {{templateSeed[{i}]}}" for i in range(len(tempDictInit["templateSeed"]))]+\
  [f"variable pdistSeed{i+1} equal {{pdistSeed[{i}]}}" for i in range(len(tempDictInit["pdistSeed"]))]+\
  [f"variable insSeed{i+1} equal {{insSeed[{i}]}}" for i in range(len(tempDictInit["insSeed"]))]
  
  myTempInit='\n'.join(myTempInit)
  #writeFileInit=pathlib.Path(os.path.join(destPath, "parametersInit.par"))
  #writeFileInit.write_text(myTempInit.format(**tempDictInit))
  
  
  
  tempDictMain={
    "dt": computeTimestep,
    "run_time": runTimeMain,
    "status_output": statusTimestep,
    "data_output": dumpTimeStep,
    "angle": angleMain,
    "rp": rpMain,
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
  ["variable dt equal {dt}"]+\
  ["variable run_time equal {run_time}"]+\
  ["variable status_output equal {status_output}"]+\
  ["variable data_output equal {data_output}"]+\
  ["variable inclination_angle equal {angle}"]+\
  ["variable rotation_period equal {rp}"]+\
  ["variable particle_size equal {maxps}"]+\
  [f"variable ym{i+1} equal {{ym[{i}]}}" for i in range(len(tempDictMain["ym"]))]+\
  [f"variable pr{i+1} equal {{pr[{i}]}}" for i in range(len(tempDictMain["pr"]))]+\
  [f"variable cr{i+1}{j+1} equal {{cr[{i}][{j}]}}" for i in range(len(tempDictMain["cr"])) for j in range(len(tempDictMain["cr"][i]))]+\
  [f"variable cf{i+1}{j+1} equal {{cf[{i}][{j}]}}" for i in range(len(tempDictMain["cf"])) for j in range(len(tempDictMain["cf"][i]))]+\
  [f"variable crf{i+1}{j+1} equal {{crf[{i}][{j}]}}" for i in range(len(tempDictMain["crf"])) for j in range(len(tempDictMain["crf"][i]))]+\
  [f"variable ced{i+1}{j+1} equal {{ced[{i}][{j}]}}" for i in range(len(tempDictMain["ced"])) for j in range(len(tempDictMain["ced"][i]))]
  
  myTempMain='\n'.join(myTempMain)
  writeFileMain=pathlib.Path(os.path.join(destPath, "parametersMain.par"))
  writeFileMain.write_text(myTempMain.format(**tempDictMain))
  
  
  
  outMeshDir=os.path.join(destPath, "meshes")
  if not os.path.exists(outMeshDir):
    os.makedirs(outMeshDir)
  
  inSTLPath=stlGitDir
  outSTLPath=outMeshDir
  
  shutil.copy(os.path.join(templateGitDir, 'initScript'), os.path.join(destPath, 'initScript'))
  shutil.copy(os.path.join(templateGitDir, 'mainScript'), os.path.join(destPath, 'mainScript'))
  shutil.copy(os.path.join(inSTLPath, 'cap.stl'), os.path.join(outSTLPath, 'cap.stl'))
  shutil.copy(os.path.join(inSTLPath, 'drum.stl'), os.path.join(outSTLPath, 'drum.stl'))
  shutil.copy(os.path.join(inSTLPath, 'inlet.stl'), os.path.join(outSTLPath, 'inlet.stl'))

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
