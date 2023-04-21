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

import itertools
import numpy as np
import pickle
import os

homeDir=os.getenv('HOME')
infoDir=os.path.join(os.environ['HOME'], "bgnnInfo")

import pickle
myf=open(os.path.join(infoDir, "hopper2Stat.pckl"), "rb")
v1=pickle.load(myf)
v2=pickle.load(myf)
v3=pickle.load(myf)
v4=pickle.load(myf)
myf.close()

trainList=list(itertools.product(list(range(30)), list(range((5+1)*1, 3001))))

val1=[]
val2=[]
val3=[]
val4=[]

ind=0
for ind in range(0,30):
  s1=np.array(v1)[np.array([x[0]==ind for x in trainList])]
  s2=np.array(v2)[np.array([x[0]==ind for x in trainList])]
  s3=np.array(v3)[np.array([x[0]==ind for x in trainList])]
  s4=np.array(v4)[np.array([x[0]==ind for x in trainList])]
  
  winInd=np.argmax((s4-s3)/(s3))
  
  val1.append(s1[winInd])
  val2.append(s2[winInd])
  val3.append(s3[winInd])
  val4.append(s4[winInd])

print(int(np.round(np.mean(val1),0)))
print(int(np.round(np.std(val1),0)))

print(int(np.round(np.mean(val2),0)))
print(int(np.round(np.std(val2),0)))

print(np.round(np.mean((np.array(val4)-np.array(val3))/np.array(val3)),3))
