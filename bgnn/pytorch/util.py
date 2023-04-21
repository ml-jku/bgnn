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

def myRandNorm(size, mean, stddev, device=device):
  if type(mean)!=torch.Tensor:
    mean=torch.from_numpy(np.array(mean)).float().to(device)
  if type(stddev)!=torch.Tensor:
    stddev=torch.from_numpy(np.array(stddev)).float().to(device)
  if mean.device!=device:
    mean=mean.to(device)
  if stddev.device!=device:
    stddev=stddev.to(device)
  return (torch.randn(size, device=device)+mean)*stddev

def myRandUniform(size, minval, maxval, device=device):
  if type(minval)!=torch.Tensor:
    minval=torch.from_numpy(np.array(minval)).float().to(device)
  if type(maxval)!=torch.Tensor:
    maxval=torch.from_numpy(np.array(maxval)).float().to(device)
  if minval.device!=device:
    minval=minval.to(device)
  if maxval.device!=device:
    maxval=maxval.to(device)
  return torch.rand(size, device=device)*(maxval-minval)+minval

def myBin(size, th, device=device):
  return (torch.rand(size, device=device)>th).float()

class KahanSum:
  def __init__(self, init, device):
    self.device=device
    self.c=torch.from_numpy(init).to(self.device)
    self.sum=torch.from_numpy(init).to(self.device)

  def add(self, nr):
    y=nr-self.c
    t=self.sum+y
    self.c=(t-self.sum)-y
    self.sum=t
  
  def readout(self):
    return self.sum+self.c

class KahanSumExtended:
  def __init__(self, init, device):
    self.device=device
    self.c1=torch.from_numpy(init).to(self.device)
    self.c2=torch.from_numpy(init).to(self.device)
    self.c3=torch.from_numpy(init).to(self.device)

  def add(self, nr):
    c1=self.c1+nr
    c2=self.c2+c1
    self.c1=c1-(c2-self.c2)
    c3=self.c3+c2
    self.c2=c2-(c3-self.c3)
    self.c3=c3
  
  def readout(self):
    return self.c1+self.c2+self.c3
  
  def getState(self):
    ret={}
    ret["c1"]=self.c1.cpu().numpy()
    ret["c2"]=self.c2.cpu().numpy()
    ret["c3"]=self.c3.cpu().numpy()
    return ret
  
  def setState(self, ret):
    self.c1=torch.from_numpy(ret["c1"]).to(self.device)
    self.c2=torch.from_numpy(ret["c2"]).to(self.device)
    self.c3=torch.from_numpy(ret["c3"]).to(self.device)

class ScalarKahanSumExtended:
  def __init__(self):
    self.c1=0.0
    self.c2=0.0
    self.c3=0.0

  def add(self, nr):
    c1=self.c1+nr
    c2=self.c2+c1
    self.c1=c1-(c2-self.c2)
    c3=self.c3+c2
    self.c2=c2-(c3-self.c3)
    self.c3=c3
  
  def readout(self):
    return self.c1+self.c2+self.c3


class KahanSumExtendedNP:
  def __init__(self):
    self.c1=np.zeros(3, dtype=np.float64)
    self.c2=np.zeros(3, dtype=np.float64)
    self.c3=np.zeros(3, dtype=np.float64)

  def add(self, nr):
    c1=self.c1+nr
    c2=self.c2+c1
    self.c1=c1-(c2-self.c2)
    c3=self.c3+c2
    self.c2=c2-(c3-self.c3)
    self.c3=c3
  
  def readout(self):
    return self.c1+self.c2+self.c3

class Statistics:
  def __init__(self, device):
    self.initiated=False
    self.sums1=[]
    self.sums2=[]
    self.elems=[]
    self.names={}
    self.sums1R=[]
    self.sums2R=[]
    self.means=[]
    self.stds=[]
    self.cache={}
    self.device=device
  
  def getState(self):
    ret={}
    ret["initiated"]=self.initiated
    ret["sums1"]=[x.getState() for x in self.sums1]
    ret["sums2"]=[x.getState() for x in self.sums2]
    ret["elems"]=self.elems.copy()
    ret["names"]=self.names.copy()
    ret["sums1R"]=[x.cpu().numpy() for x in self.sums1R]
    ret["sums2R"]=[x.cpu().numpy() for x in self.sums2R]
    ret["means"]=[x.cpu().numpy() for x in self.means]
    ret["stds"]=[x.cpu().numpy() for x in self.stds]
    return ret
  
  def setState(self, ret):
    self.initiated=ret["initiated"]
    for i in range(0, len(ret["sums1"])):
      self.sums1.append(KahanSumExtended(np.zeros_like(ret["sums1"][i]["c1"]), device=self.device))
      self.sums1[-1].setState(ret["sums1"][i])
    for i in range(0, len(ret["sums2"])):
      self.sums2.append(KahanSumExtended(np.zeros_like(ret["sums2"][i]["c1"]), device=self.device))
      self.sums2[-1].setState(ret["sums2"][i])
    self.elems=ret["elems"].copy()
    self.names=ret["names"].copy()
    for i in range(0, len(ret["sums1R"])):
      self.sums1R.append(torch.from_numpy(ret["sums1R"][i]).to(self.device))
    for i in range(0, len(ret["sums2R"])):
      self.sums2R.append(torch.from_numpy(ret["sums2R"][i]).to(self.device))
    for i in range(0, len(ret["means"])):
      self.means.append(torch.from_numpy(ret["means"][i]).to(self.device))
    for i in range(0, len(ret["stds"])):
      self.stds.append(torch.from_numpy(ret["stds"][i]).to(self.device))
  
  def add(self, name, dim):
    self.names[name]=len(self.elems)
    self.sums1.append(KahanSumExtended(np.zeros((1, *dim), dtype=np.float32), device=self.device))
    self.sums2.append(KahanSumExtended(np.zeros((1, *dim), dtype=np.float32), device=self.device))
    self.sums1R.append(None)
    self.sums2R.append(None)
    self.elems.append(0)
  
  def track(self, name, mat):
    if not self.initiated:
      self.add(name, tuple(mat.shape[1:]))
    ind=self.names[name]
    self.sums1[ind].add(torch.sum(mat, 0, keepdims=True))
    self.sums2[ind].add(torch.sum(mat**2, 0, keepdims=True))
    self.elems[ind]=self.elems[ind]+mat.shape[0]
  
  def trackZeros(self, name, statShape, addNr):
    if not self.initiated:
      self.add(name, statShape)
    ind=self.names[name]
    self.elems[ind]=self.elems[ind]+addNr
  
  def trackE(self, name, mat, epochNr, lockStat):
    if epochNr==0 and (not lockStat):
      self.track(name, mat)
  
  def trackZerosE(name, statShape, addNr, epochNr, lockStat):
    if epochNr==0 and (not lockStat):
      self.trackZeros(name, statShape, addNr, epochNr)  
  
  def endTrack(self):
    self.initiated=True
    self.means=[]
    self.stds=[]
    self.cache={}
    for ind in range(0, len(self.elems)):
      self.sums1R[ind]=self.sums1[ind].readout()
      self.sums2R[ind]=self.sums2[ind].readout()
      self.means.append(self.sums1R[ind]/float(self.elems[ind]))
      self.stds.append(torch.sqrt((self.sums2R[ind]/float(self.elems[ind]))-self.means[-1]**2))
      self.stds[-1]=torch.where(self.stds[-1]==0.0, torch.tensor(1.0,device=self.device), self.stds[-1])
  
  def getMean(self, name):
    ind=self.names[name]
    return self.means[ind]
  
  def getStd(self, name):
    ind=self.names[name]
    return self.stds[ind]
  
  def getPairStat(self, name1, name2):
    if (name1, name2) in self.cache:
      return self.cache[(name1, name2)]
    else:
      ind1=self.names[name1]
      ind2=self.names[name2]
      mymean=(self.sums1R[ind1]+self.sums1R[ind2])/(float(self.elems[ind1]+self.elems[ind2]))
      mystd=torch.sqrt(((self.sums2R[ind1]+self.sums2R[ind2])/float((self.elems[ind1]+self.elems[ind2])))-mymean**2)
      mystd=torch.where(mystd==0.0, torch.tensor(1.0,device=self.device), mystd)
      #mystd=tf.where(tf.math.is_nan(mystd), 1.0, mystd)
      self.cache[(name1, name2)]=(mymean, mystd)
      return self.cache[(name1, name2)]



def normParS(rpsetting, bstat, lenName1, lenName2, simpleScalarMeanPar, simpleScalarStdPar):
  if rpsetting[2]==0:
    meanLen=bstat.getMean(lenName1)
    stdLen=bstat.getStd(lenName1)
  elif rpsetting[2]==1:
    meanLen, stdLen=bstat.getPairStat(lenName1, lenName2)
  
  if rpsetting[0]==1:
    meanPar=meanLen*0.0
  elif rpsetting[0]==2:
    meanPar=meanLen
  elif rpsetting[0]==4:
    meanPar=simpleScalarMeanPar
  
  if rpsetting[1]==1:
    stdPar=stdLen
  elif rpsetting[1]==2:
    stdPar=stdLen
  elif rpsetting[1]==3:
    stdPar=simpleScalarStdPar
  elif rpsetting[1]==4:
    stdPar=simpleScalarStdPar
  
  meanP=meanPar
  stdP=stdPar
  if rpsetting[3]==0:
    meanW=meanPar
    stdW=stdPar
  elif rpsetting[3]==1:
    meanW=meanLen*0.0
    stdW=stdPar
  
  return meanP, stdP, meanW, stdW


def normParV(rpsetting, bstat, vecName1, vecName2, lenName1, lenName2, simpleVectorMeanPar, simpleScalarStdPar, simpleVectorStdPar):
  if rpsetting[2]==0:
    meanVec=bstat.getMean(vecName1)
    meanLen=bstat.getMean(lenName1)
    stdVec=bstat.getStd(vecName1)
    stdLen=bstat.getStd(lenName1)
  elif rpsetting[2]==1:
    meanVec, stdVec=bstat.getPairStat(vecName1, vecName2)
    meanLen, stdLen=bstat.getPairStat(lenName1, lenName2)
  
  if rpsetting[0]==1:
    meanPar=meanVec*0.0
  elif rpsetting[0]==2:
    meanPar=meanVec
  elif rpsetting[0]==4:
    meanPar=simpleVectorMeanPar
  
  if rpsetting[1]==1:
    stdPar=stdLen
  elif rpsetting[1]==2:
    stdPar=stdVec
  elif rpsetting[1]==3:
    stdPar=simpleScalarStdPar
  elif rpsetting[1]==4:
    stdPar=simpleVectorStdPar
  
  meanP=meanPar
  stdP=stdPar
  if rpsetting[3]==0:
    meanW=meanPar
    stdW=stdPar
  elif rpsetting[3]==1:
    meanW=meanVec*0.0
    stdW=stdPar
  
  return meanP, stdP, meanW, stdW

class Normalize:
  def __init__(self):
    self.parSN=[]
    self.parVN=[]
    self.parSE=[]
    self.parVE=[]
  
  def registerSN(self, *args):
    self.parSN.append(args)
  
  def registerVN(self, *args):
    self.parVN.append(args)
    
  def registerSE(self, *args):
    self.parSE.append(args)
  
  def registerVE(self, *args):
    self.parVE.append(args)
  
  def normalize(self, particleNodeDataList, wallNodeDataList, particleEdgeDataList, wallEdgeDataList):
    for i in range(len(self.parSN)):
      meanValP, stdValP, meanValW, stdValW=self.parSN[i][0](*self.parSN[i][1:-2])
      pten=particleNodeDataList[self.parSN[i][-2]]
      wten=wallNodeDataList[self.parSN[i][-1]]
      particleNodeDataList[self.parSN[i][-2]]=torch.reshape((pten-meanValP)/stdValP,(pten.shape[0],np.prod(pten.shape[1:])))
      wallNodeDataList[self.parSN[i][-1]]=torch.reshape((wten-meanValW)/stdValW,(wten.shape[0],np.prod(wten.shape[1:])))
    for i in range(len(self.parSE)):
      meanValP, stdValP, meanValW, stdValW=self.parSE[i][0](*self.parSE[i][1:-2])
      pten=particleEdgeDataList[self.parSE[i][-2]]
      wten=wallEdgeDataList[self.parSE[i][-1]]
      particleEdgeDataList[self.parSE[i][-2]]=torch.reshape((pten-meanValP)/stdValP,(pten.shape[0],np.prod(pten.shape[1:])))
      wallEdgeDataList[self.parSE[i][-1]]=torch.reshape((wten-meanValW)/stdValW,(wten.shape[0],np.prod(wten.shape[1:])))
    for i in range(len(self.parVN)):
      meanValP, stdValP, meanValW, stdValW=self.parVN[i][0](*self.parVN[i][1:-2])
      pten=particleNodeDataList[self.parVN[i][-2]]
      wten=wallNodeDataList[self.parVN[i][-1]]
      particleNodeDataList[self.parVN[i][-2]]=torch.reshape((pten-meanValP)/stdValP,(pten.shape[0],np.prod(pten.shape[1:])))
      wallNodeDataList[self.parVN[i][-1]]=torch.reshape((wten-meanValW)/stdValW,(wten.shape[0],np.prod(wten.shape[1:])))
    for i in range(len(self.parVE)):
      meanValP, stdValP, meanValW, stdValW=self.parVE[i][0](*self.parVE[i][1:-2])
      pten=particleEdgeDataList[self.parVE[i][-2]]
      wten=wallEdgeDataList[self.parVE[i][-1]]
      particleEdgeDataList[self.parVE[i][-2]]=torch.reshape((pten-meanValP)/stdValP,(pten.shape[0],np.prod(pten.shape[1:])))
      wallEdgeDataList[self.parVE[i][-1]]=torch.reshape((wten-meanValW)/stdValW,(wten.shape[0],np.prod(wten.shape[1:])))
