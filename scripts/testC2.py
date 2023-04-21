#Source codes of the publication "Boundary Graph Neural Networks for 3D Simulations"
#Copyright (C) 2023 Andreas Mayr

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

#TApp. C.2

# fexp1: 0
# rp["multNormV"]=0.0
# rp["options"]="featureOptions1.py"
# useDist
# "== no NV"

# fexp2: 1
# rp["multNormV"]=10.0
# rp["options"]="featureOptions2.py"
# useDist
# "== single NV"

# fexp3: 2
# rp["multNormV"]=10.0
# rp["options"]="featureOptions1.py"
# useDist
# "== both NV"

# fexp4: 3
# rp["multNormV"]=0.0
# rp["options"]="featureOptions1.py"
# useInvDist2
# "== no NV"

# fexp5: 4
# rp["multNormV"]=10.0
# rp["options"]="featureOptions2.py"
# useInvDist2
# "== single NV"

# fexp6: 5
# rp["multNormV"]=10.0
# rp["options"]="featureOptions1.py"
# useInvDist2
# "== both NV"

# fexp7: 6
# rp["multNormV"]=0.0
# rp["options"]="featureOptions1.py"
# useInvDist
# "== no NV"

# fexp8: 7
# rp["multNormV"]=10.0
# rp["options"]="featureOptions2.py"
# useInvDist
# "== single NV"

# fexp9: 8
# rp["multNormV"]=10.0
# rp["options"]="featureOptions1.py"
# useInvDist
# "== both NV"



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
import os
import os.path
import datetime
import json
import time
import pickle
import scipy
import scipy.stats

problem="hopper"
experiment="runs0"
startTime=6
conInd=[1,2,3,4,5,6,7,8,9]
plotSequences=[0,1,2,3,4,35,36,37,38,39]
measure="emd"


trainMean=[]
testMean=[]
trainSd=[]
testSd=[]
trainSamples=[]
testSamples=[]
considerDirs=[x for x in np.sort(os.listdir("/system/user/mayr-data/BGNNRuns/evaluations/"+problem+"/"+experiment))]

for i in range(len(conInd)):
  considerDir=[x for x in  considerDirs if x.startswith("fexp"+str(conInd[i]))][0]
  evalDir=os.path.join("/system/user/mayr-data/BGNNRuns/evaluations/"+problem+"/"+experiment, considerDir)
  
  emd=[]
  for j in range(len(plotSequences)):
    f=open(os.path.join(evalDir, "emd_"+str(plotSequences[j])+"_"+str(startTime)+".pckl"), "rb")
    emdRes=pickle.load(f)
    f.close()
    emd.append(emdRes)
  trainMean.append(np.array(emd[0:5]).mean(0))
  testMean.append(np.array(emd[5:10]).mean(0))
  trainSd.append(np.array(emd[0:5]).std(0))
  testSd.append(np.array(emd[5:10]).std(0))
  
  trainSamples.append(np.concatenate(emd[0:5]))
  testSamples.append(np.concatenate(emd[5:10]))

pmat=np.ones((len(trainSamples), len(trainSamples)))
for i in range(0, len(trainSamples)):
  for j in range(0, len(trainSamples)):
    print(scipy.stats.wilcoxon(trainSamples[i], trainSamples[j], alternative="less", zero_method="zsplit").pvalue)
    pmat[i,j]=scipy.stats.wilcoxon(trainSamples[i], trainSamples[j], alternative="less", zero_method="zsplit").pvalue
winnerInd=np.argmax(-np.log(pmat).mean(1)) #not used winner Index (just out of interest to analyse the whole procedure)

#Hyperparameter selection (on training samples, as not well trained models can often directly be observed on training trajectories)
ind0=0+3*np.argmax(-np.log(pmat[0:pmat.shape[0]:3,0:pmat.shape[0]:3]).mean(1))
ind1=1+3*np.argmax(-np.log(pmat[1:pmat.shape[0]:3,1:pmat.shape[0]:3]).mean(1))
ind2=2+3*np.argmax(-np.log(pmat[2:pmat.shape[0]:3,2:pmat.shape[0]:3]).mean(1))

np.array([ind0, ind1, ind2])


pmatTest=np.ones((len(testSamples), len(testSamples)))
for i in range(0, len(testSamples)):
  for j in range(0, len(testSamples)):
    print(scipy.stats.wilcoxon(testSamples[i], testSamples[j], alternative="less", zero_method="zsplit").pvalue)
    pmatTest[i,j]=scipy.stats.wilcoxon(testSamples[i], testSamples[j], alternative="less", zero_method="zsplit").pvalue
winnerIndTest=np.argmax(-np.log(pmatTest[np.array([ind0, ind1, ind2])][:,np.array([ind0, ind1, ind2])]).mean(1))

print(np.array(considerDirs)[np.array([ind0, ind1, ind2])])

print(pmatTest[np.array([ind0, ind1, ind2])][:,np.array([ind0, ind1, ind2])])
print(winnerIndTest)
print(np.format_float_scientific(testSamples[ind0].mean(), precision=2))
print(np.format_float_scientific(testSamples[ind1].mean(), precision=2))
print(np.format_float_scientific(testSamples[ind2].mean(), precision=2))
print(np.format_float_scientific(testSamples[ind0].std(), precision=2))
print(np.format_float_scientific(testSamples[ind1].std(), precision=2))
print(np.format_float_scientific(testSamples[ind2].std(), precision=2))



pmatTrain=np.ones((len(trainSamples), len(trainSamples)))
for i in range(0, len(trainSamples)):
  for j in range(0, len(trainSamples)):
    print(scipy.stats.wilcoxon(trainSamples[i], trainSamples[j], alternative="less", zero_method="zsplit").pvalue)
    pmatTrain[i,j]=scipy.stats.wilcoxon(trainSamples[i], trainSamples[j], alternative="less", zero_method="zsplit").pvalue
winnerIndTrain=np.argmax(-np.log(pmatTrain[np.array([ind0, ind1, ind2])][:,np.array([ind0, ind1, ind2])]).mean(1))

print(np.array(considerDirs)[np.array([ind0, ind1, ind2])])

print(pmatTrain[np.array([ind0, ind1, ind2])][:,np.array([ind0, ind1, ind2])])
print(winnerIndTrain)
print(np.format_float_scientific(trainSamples[ind0].mean(), precision=2))
print(np.format_float_scientific(trainSamples[ind1].mean(), precision=2))
print(np.format_float_scientific(trainSamples[ind2].mean(), precision=2))
print(np.format_float_scientific(trainSamples[ind0].std(), precision=2))
print(np.format_float_scientific(trainSamples[ind1].std(), precision=2))
print(np.format_float_scientific(trainSamples[ind2].std(), precision=2))
