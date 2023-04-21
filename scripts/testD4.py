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

#TApp. D.4

import sys
import numpy as np
import scipy.stats

gtAll=[]
predAll=[]
for myind in range(40,70):
  gt=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs3/model/gtp_"+str(myind)+"_6.npy")
  pred=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs3/model/predp_"+str(myind)+"_6.npy")
  gtAll.append(gt[2500])
  predAll.append(pred[2500])

prop=[float(np.sum(predAll[x][:,2]<0))/float(predAll[x].shape[0]) for x in range(0,30)]

ood=prop[0:15]
nonood=prop[15:30]

print(np.array(ood).mean())
print(np.array(ood).std())
print(np.array(nonood).mean())
print(np.array(nonood).std())
print(scipy.stats.mannwhitneyu(nonood, ood, alternative='greater').pvalue)

ood1=prop[0:15]
nonood1=prop[15:30]


print("---")


gtAll=[]
predAll=[]
for myind in range(40,70):
  gt=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs4/model/gtp_"+str(myind)+"_6.npy")
  pred=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs4/model/predp_"+str(myind)+"_6.npy")
  gtAll.append(gt[2500])
  predAll.append(pred[2500])

prop=[float(np.sum(predAll[x][:,2]<0))/float(predAll[x].shape[0]) for x in range(0,30)]

ood=prop[0:15]
nonood=prop[15:30]

print(np.array(ood).mean())
print(np.array(ood).std())
print(np.array(nonood).mean())
print(np.array(nonood).std())
print(scipy.stats.mannwhitneyu(nonood, ood, alternative='greater').pvalue)

ood2=prop[0:15]
nonood2=prop[15:30]

print(scipy.stats.mannwhitneyu(ood2, ood1, alternative='greater').pvalue)