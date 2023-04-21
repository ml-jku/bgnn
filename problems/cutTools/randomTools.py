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

def rellipsoid(pointData, posData, rtype, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  x3=pointData[:,2]
  remPoints=20000
  trialNr=0
  while (remPoints>maxPoints) or (remPoints<minPoints):
    #rtype=int(np.random.uniform(0.0,1.0)>=0.5)
    rposx=np.random.uniform(-0.5,0.5)
    rposy=np.random.uniform(-0.5,0.5)
    rscalex=np.random.uniform(0.8,2.0)
    rscaley=np.random.uniform(0.8,2.0)
    rscalez=np.random.uniform(0.1,1.5)
    ralpha=np.random.normal(0, 0.25*math.pi/2)
    cut0=x3<ellipsoid(pointData, posData, type=rtype, circle=False, posx=rposx, posy=rposy, scalex=rscalex, scaley=rscaley, scalez=rscalez, alpha=ralpha)
    remPoints=np.sum(cut0)
    par={"rtype": rtype, "rposx": rposx, "rposy": rposy, "rscalex": rscalex, "rscaley": rscaley, "rscalez": rscalez, "ralpha": ralpha}
    trialNr=trialNr+1
    if trialNr==nrTrials:
      trialNr=0
      minPoints=int(minPoints/minFactor)
      maxPoints=maxPoints*maxFactor
  return(cut0, par)

def rgaussian(pointData, posData, rtype, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  x3=pointData[:,2]
  remPoints=20000
  trialNr=0
  while (remPoints>maxPoints) or (remPoints<minPoints):
    #rtype=int(np.random.uniform(0.0,1.0)>=0.5)
    rposx=np.random.uniform(-0.5,0.5)
    rposy=np.random.uniform(-0.5,0.5)
    rscalex=np.random.uniform(0.8,2.0)
    rscaley=np.random.uniform(0.8,2.0)
    rscalez=np.random.uniform(0.8,1.5)
    ralpha=np.random.normal(0, 0.25*math.pi/2)
    cut0=x3<gaussian(pointData, posData, type=rtype, circle=False, posx=rposx, posy=rposy, scalex=rscalex, scaley=rscaley, scalez=rscalez, alpha=ralpha)
    remPoints=np.sum(cut0)
    par={"rtype": rtype, "rposx": rposx, "rposy": rposy, "rscalex": rscalex, "rscaley": rscaley, "rscalez": rscalez, "ralpha": ralpha}
    trialNr=trialNr+1
    if trialNr==nrTrials:
      trialNr=0
      minPoints=int(minPoints/minFactor)
      maxPoints=maxPoints*maxFactor
  return(cut0, par)

def rlinXZ(pointData, posData, rtype, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  x3=pointData[:,2]
  remPoints=20000
  trialNr=0
  while (remPoints>maxPoints) or (remPoints<minPoints):
    #rtype=int(np.random.uniform(0.0,1.0)>=0.5)
    if rtype>0.5:
      rbeta=np.random.normal(math.pi, 0.25*(math.pi/4.0))
    else:
      rbeta=np.random.normal(0, 0.25*(math.pi/4.0))
    rd=np.random.uniform(-1.0,1.0)
    ralpha=np.random.uniform(-math.pi/4,math.pi/4)
    #rbeta=np.random.uniform(-math.pi/4,math.pi/4)
    cut0=x3<linXZ(pointData, posData, d=rd, alpha=ralpha, beta=rbeta)
    remPoints=np.sum(cut0)
    par={"rtype": rtype, "rd": rd, "ralpha": ralpha, "rbeta": rbeta}
    trialNr=trialNr+1
    if trialNr==nrTrials:
      trialNr=0
      minPoints=int(minPoints/minFactor)
      maxPoints=maxPoints*maxFactor
  return(cut0, par)

def rlinYZ(pointData, posData, rtype, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  x3=pointData[:,2]
  remPoints=20000
  trialNr=0
  while (remPoints>maxPoints) or (remPoints<minPoints):
    if rtype>0.5:
      rbeta=np.random.normal(math.pi, 0.25*(math.pi/4.0))
    else:
      rbeta=np.random.normal(0, 0.25*(math.pi/4.0))
    #rtype=int(np.random.uniform(0.0,1.0)>=0.5)
    rd=np.random.uniform(-1.0,1.0)
    ralpha=np.random.uniform(-math.pi/4,math.pi/4)
    #rbeta=np.random.uniform(-math.pi/4,math.pi/4)
    cut0=x3<linYZ(pointData, posData, d=rd, alpha=ralpha, beta=rbeta)
    remPoints=np.sum(cut0)
    par={"rtype": rtype, "rd": rd, "ralpha": ralpha, "rbeta": rbeta}
    trialNr=trialNr+1
    if trialNr==nrTrials:
      trialNr=0
      minPoints=int(minPoints/minFactor)
      maxPoints=maxPoints*maxFactor
  return(cut0, par)


def rlinXY(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  x3=pointData[:,2]
  remPoints=20000
  trialNr=0
  while (remPoints>maxPoints) or (remPoints<minPoints):
    rd=np.random.uniform(0.0,1.0)
    ralpha=np.random.uniform(-math.pi/4,math.pi/4)
    rbeta=np.random.uniform(-math.pi/4,math.pi/4)
    cut0=x3<linXY(pointData, posData, d=rd, alpha=ralpha, beta=rbeta)
    remPoints=np.sum(cut0)
    par={"rd": rd, "ralpha": ralpha, "rbeta": rbeta}
    trialNr=trialNr+1
    if trialNr==nrTrials:
      trialNr=0
      minPoints=int(minPoints/minFactor)
      maxPoints=maxPoints*maxFactor
  return(cut0, par)

def rcombMax(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  x3=pointData[:,2]
  remPoints=20000
  trialNr=0
  while (remPoints>maxPoints) or (remPoints<minPoints):
    rpropWalls=np.random.uniform(0.001, 1.0)
    rpropBottom=np.random.uniform(-math.pi/4,math.pi/4)
    cut0=x3<combMax(pointData, posData, rpropWalls, rpropBottom)
    remPoints=np.sum(cut0)
    par={"rtype": 1, "rpropWalls": rpropWalls, "rpropBottom": rpropBottom}
    trialNr=trialNr+1
    if trialNr==nrTrials:
      trialNr=0
      minPoints=int(minPoints/minFactor)
      maxPoints=maxPoints*maxFactor
  return(cut0, par)

def rcombMin(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  x3=pointData[:,2]
  remPoints=20000
  trialNr=0
  while (remPoints>maxPoints) or (remPoints<minPoints):
    rpropWalls=np.random.uniform(0.001, 1.0)
    rpropBottom=np.random.uniform(-math.pi/4,math.pi/4)
    cut0=x3<combMax(pointData, posData, rpropWalls, rpropBottom)
    remPoints=np.sum(cut0)
    par={"rtype": 0, "rpropWalls": rpropWalls, "rpropBottom": rpropBottom}
    trialNr=trialNr+1
    if trialNr==nrTrials:
      trialNr=0
      minPoints=int(minPoints/minFactor)
      maxPoints=maxPoints*maxFactor
  return(cut0, par)

def rcomb(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  rtype=int(np.random.uniform(0.0,1.0)>=0.5)
  if rtype>0.5:
    return rcombMax(pointData, posData, minPoints, maxPoints, nrTrials, minFactor, maxFactor)
  else:
    return rcombMin(pointData, posData, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand0(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rellipsoid(pointData, posData, 0, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand1(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rellipsoid(pointData, posData, 1, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand2(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rgaussian(pointData, posData, 0, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand3(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rgaussian(pointData, posData, 1, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand4(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rlinXZ(pointData, posData, 0, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand5(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rlinXZ(pointData, posData, 1, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand6(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rlinYZ(pointData, posData, 0, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand7(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rlinYZ(pointData, posData, 1, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand8(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rlinXY(pointData, posData, minPoints, maxPoints, nrTrials, minFactor, maxFactor)

def rand9(pointData, posData, minPoints=1000, maxPoints=8000, nrTrials=10, minFactor=4.0, maxFactor=1.21):
  return rcomb(pointData, posData, minPoints, maxPoints, nrTrials, minFactor, maxFactor)