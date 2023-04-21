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

def qform1(x1, x2,   r1, r2, r3,   d1, d2, d3,   v11, v12, v13, v21, v22, v23, v31, v32, v33):
  ret=(d2*r1*v21*v23 + d2*r2*v22*v23 + d2*r3*v23**2 + d3*r1*v31*v33 + d3*r2*v32*v33 + d3*r3*v33**2 - d2*v21*v23*x1 - d3*v31*v33*x1 - d2*v22*v23*x2 - d3*v32*v33*x2 + d1*v13*(r1*v11 + r2*v12 + r3*v13 - v11*x1 - v12*x2) - np.sqrt(4*(d1*v13*(r1*v11 + r2*v12 + r3*v13 - v11*x1 - v12*x2) + d2*v23*(r1*v21 + r2*v22 + r3*v23 - v21*x1 - v22*x2) + d3*v33*(r1*v31 + r2*v32 + r3*v33 - v31*x1 - v32*x2))**2 - 4*(d1*v13**2 + d2*v23**2 + d3*v33**2)*(-1 + d1*(r1*v11 + r2*v12 + r3*v13 - v11*x1 - v12*x2)**2 + d2*(r1*v21 + r2*v22 + r3*v23 - v21*x1 - v22*x2)**2 + d3*(r1*v31 + r2*v32 + r3*v33 - v31*x1 - v32*x2)**2))/2.)/(d1*v13**2 + d2*v23**2 + d3*v33**2)
  ret[np.isnan(ret)]=np.inf
  return ret

def qform2(x1, x2,   r1, r2, r3,   d1, d2, d3,   v11, v12, v13, v21, v22, v23, v31, v32, v33):
  ret=(d2*r1*v21*v23 + d2*r2*v22*v23 + d2*r3*v23**2 + d3*r1*v31*v33 + d3*r2*v32*v33 + d3*r3*v33**2 - d2*v21*v23*x1 - d3*v31*v33*x1 - d2*v22*v23*x2 - d3*v32*v33*x2 + d1*v13*(r1*v11 + r2*v12 + r3*v13 - v11*x1 - v12*x2) + np.sqrt(4*(d1*v13*(r1*v11 + r2*v12 + r3*v13 - v11*x1 - v12*x2) + d2*v23*(r1*v21 + r2*v22 + r3*v23 - v21*x1 - v22*x2) + d3*v33*(r1*v31 + r2*v32 + r3*v33 - v31*x1 - v32*x2))**2 - 4*(d1*v13**2 + d2*v23**2 + d3*v33**2)*(-1 + d1*(r1*v11 + r2*v12 + r3*v13 - v11*x1 - v12*x2)**2 + d2*(r1*v21 + r2*v22 + r3*v23 - v21*x1 - v22*x2)**2 + d3*(r1*v31 + r2*v32 + r3*v33 - v31*x1 - v32*x2)**2))/2.)/(d1*v13**2 + d2*v23**2 + d3*v33**2)
  ret[np.isnan(ret)]=np.inf
  return ret

def Ra(alpha):
  return np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])

def ellipsoid(pointData, posData, type=0, circle=False, posx=0.0, posy=0.0, scalex=1.0, scaley=1.0, scalez=0.5, alpha=0):
  #type: 0/1
  #posx, posy: any number; [-1, 1] is the "approximate" area
  #scalex, scaley, scalez: any positive number (0, infinity), 1.0 is the "approximate" area
  #alpha=0..2Pi
  x1=pointData[:,0]
  x2=pointData[:,1]
  mx, my, mz, lx, ly, lz=posData
  
  cx=mx+lx/2.0
  cy=my+ly/2.0
  cz=mz
  
  dimx=scalex*lx
  dimy=scaley*ly
  if circle:
    dimx=min(dimx, dimy)
    dimy=dimx
  dimz=scalez*lz
  
  r1=posx*(lx/2.0)+cx
  r2=posy*(ly/2.0)+cy
  if type==0:
    r3=pointData.max(0)[2]
  elif type==1:
    r3=cz
  
  v11=1.0
  v12=0.0
  v13=0.0
  v21=0.0
  v22=1.0
  v23=0.0
  v31=0.0
  v32=0.0
  v33=1.0
  
  d1=(1.0/(dimx/2.0)**2)
  d2=(1.0/(dimy/2.0)**2)
  d3=(1.0/(dimz)**2)
  
  res=np.dot(Ra(alpha), np.array([v11, v12]))
  v11=res[0]
  v12=res[1]
  res=np.dot(Ra(alpha), np.array([v21, v22]))
  v21=res[0]
  v22=res[1]
  
  if type==0:
    myq=qform1(x1, x2, r1, r2, r3, d1, d2, d3, v11, v12, v13, v21, v22, v23, v31, v32, v33)
  elif type==1:
    myq=qform2(x1, x2, r1, r2, r3, d1, d2, d3, v11, v12, v13, v21, v22, v23, v31, v32, v33)
  return myq

def gaussian(pointData, posData, type=0, circle=False, posx=0.0, posy=0.0, scalex=1.0, scaley=1.0, scalez=0.5, alpha=0):
  #type: 0/1
  #posx, posy: any number; [-1, 1] is the "approximate" area
  #scalex, scaley, scalez: any positive number (0, infinity), 1.0 is the "approximate" area
  #alpha=0..2Pi
  x1=pointData[:,0]
  x2=pointData[:,1]
  mx, my, mz, lx, ly, lz=posData
  
  cx=mx+lx/2.0
  cy=my+ly/2.0
  cz=mz
  
  dimx=scalex*lx
  dimy=scaley*ly
  if circle:
    dimx=min(dimx, dimy)
    dimy=dimx
  dimz=scalez*lz
  
  r1=posx*(lx/2.0)+cx
  r2=posy*(ly/2.0)+cy
  if type==0:
    r3=pointData.max(0)[2]
  elif type==1:
    r3=cz
  
  v11=1.0
  v12=0.0
  v21=0.0
  v22=1.0
  
  res=np.dot(Ra(alpha), np.array([v11, v12]))
  v11=res[0]
  v12=res[1]
  res=np.dot(Ra(alpha), np.array([v21, v22]))
  v21=res[0]
  v22=res[1]
  
  d1=(((dimx)/2.0)/3.0)**2
  d2=(((dimy)/2.0)/3.0)**2
  
  cov2=np.array([[d1*v11*v11+d2*v21*v21, d1*v11*v12+d2*v21*v22], [d1*v11*v12+d2*v21*v22, d1*v12*v12+d2*v22*v22]])
  d3=(1.0/scipy.stats.multivariate_normal.pdf(np.array([r1, r2]), mean=np.array([r1, r2]), cov=cov2))*dimz
  
  if type==0:
    myq=r3-d3*scipy.stats.multivariate_normal.pdf(pointData[:,0:2], mean=np.array([r1, r2]), cov=cov2)
  elif type==1:
    myq=r3+d3*scipy.stats.multivariate_normal.pdf(pointData[:,0:2], mean=np.array([r1, r2]), cov=cov2)
  return myq

def Rx(alpha):
  return np.array([[1,0,0],[0, np.cos(alpha), -np.sin(alpha)],[0, np.sin(alpha), np.cos(alpha)]])

def Ry(alpha):
  return np.array([[np.cos(alpha), 0, np.sin(alpha)],[0,1,0],[-np.sin(alpha), 0, np.cos(alpha)]])

def Rz(alpha):
  return np.array([[np.cos(alpha), -np.sin(alpha), 0],[np.sin(alpha), np.cos(alpha), 0],[0, 0, 1]])

def linXZ(pointData, posData, d, alpha, beta):
  #d=-1..+1
  #alpha=-math.pi/4..math.pi/4
  #beta=-math.pi/4..math.pi/4
  x1=pointData[:,0]
  x2=pointData[:,1]
  mx, my, mz, lx, ly, lz=posData
  
  cx=mx+lx/2.0
  cy=my+ly/2.0
  cz=mz
  
  dimd=(ly/2.0)
  mind=cy
  nvec=np.dot(Rz(beta), np.dot(Rx(-alpha), np.array([0.0,1.0,0.0])))
  mind=nvec[0]*cx+nvec[1]*cy+nvec[2]*cz
  #if mind<0:
  #  mind=mind*(-1)
  return (d*dimd*nvec[1]+mind-nvec[0]*pointData[:,0]-nvec[1]*pointData[:,1])/nvec[2]

def linYZ(pointData, posData, d, alpha, beta):
  #d=-1..+1
  #alpha=-math.pi/4..math.pi/4
  #beta=-math.pi/4..math.pi/4
  x1=pointData[:,0]
  x2=pointData[:,1]
  mx, my, mz, lx, ly, lz=posData
  
  cx=mx+lx/2.0
  cy=my+ly/2.0
  cz=mz
  
  dimd=(lx/2.0)
  mind=cx
  nvec=np.dot(Rz(beta), np.dot(Ry(alpha), np.array([1.0,0.0,0.0])))
  mind=nvec[0]*cx+nvec[1]*cy+nvec[2]*cz
  #if mind<0:
  #  mind=mind*(-1)
  return (d*dimd*nvec[0]+mind-nvec[0]*pointData[:,0]-nvec[1]*pointData[:,1])/nvec[2]

def linXY(pointData, posData, d, alpha, beta):
  #d=0..1
  #alpha=-math.pi/4..math.pi/4
  #beta=-math.pi/4..math.pi/4
  x1=pointData[:,0]
  x2=pointData[:,1]
  mx, my, mz, lx, ly, lz=posData
  
  cx=mx+lx/2.0
  cy=my+ly/2.0
  cz=mz
  
  dimd=lz
  mind=mz
  nvec=np.dot(Rz(beta), np.dot(Rx(alpha), np.array([0.0,0.0,1.0])))
  mind=nvec[0]*cx+nvec[1]*cy+nvec[2]*cz
  #if mind<0:
  #  mind=mind*(-1)
  return (d*dimd*nvec[2]+mind-nvec[0]*pointData[:,0]-nvec[1]*pointData[:,1])/nvec[2]

# def combMax(pointData, posData, propWalls, propBottom):
  # #propWalls...0.2-0.4
  # c1=linXZ(pointData, posData, -0.5, -propWalls*math.pi/4, 0.0)
  # c0=c1
  # c2=linYZ(pointData, posData, -0.5, -propWalls*math.pi/4, 0.0)
  # c0=np.maximum(c0, c2)
  # c3=linXZ(pointData, posData, -0.5, -propWalls*math.pi/4, math.pi)
  # c0=np.maximum(c0, c3)
  # c4=linYZ(pointData, posData, -0.5, -propWalls*math.pi/4, math.pi)
  # c0=np.maximum(c0, c4)
  # c5=linXY(pointData, posData, 0.1, -propBottom*math.pi/4, 0.0)
  # c0=np.maximum(c0, c5)
  # return c0

# def combMin(pointData, posData, propWalls, propBottom):
  # #propWalls...0.01-0.2
  # c1=linXZ(pointData, posData, -0.5, propWalls*math.pi/4, 0.0)
  # c0=c1
  # c2=linYZ(pointData, posData, -0.5, propWalls*math.pi/4, 0.0)
  # c0=np.minimum(c0, c2)
  # c3=linXZ(pointData, posData, -0.5, propWalls*math.pi/4, math.pi)
  # c0=np.minimum(c0, c3)
  # c4=linYZ(pointData, posData, -0.5, propWalls*math.pi/4, math.pi)
  # c0=np.minimum(c0, c4)
  # c5=linXY(pointData, posData, 0.3, -propBottom*math.pi/4, 0.0)
  # c0=np.minimum(c0, c5)
  # return c0

# def combMax(pointData, posData, propWalls, propBottom):
  # #propWalls...0.0-1.0
  # c1=linXZ(pointData, posData, -1.0-1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, 0.0)
  # c0=c1
  # c2=linYZ(pointData, posData, -1.0-1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, 0.0)
  # c0=np.maximum(c0, c2)
  # c3=linXZ(pointData, posData, -1.0-1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, math.pi)
  # c0=np.maximum(c0, c3)
  # c4=linYZ(pointData, posData, -1.0-1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, math.pi)
  # c0=np.maximum(c0, c4)
  # c5=linXY(pointData, posData, 0.1, -propBottom*math.pi/4, 0.0)
  # c0=np.maximum(c0, c5)
  # return c0

def combMax(pointData, posData, propWalls, propBottom):
  #propWalls...0.0-1.0
  c1=linXZ(pointData, posData, -1.0-1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, 0.0)
  c0=c1
  c2=linYZ(pointData, posData, -1.0-1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, 0.0)
  c0=np.maximum(c0, c2)
  c3=linXZ(pointData, posData, 1.0+1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, math.pi)
  c0=np.maximum(c0, c3)
  c4=linYZ(pointData, posData, 1.0+1.5*math.tan(-propWalls*math.pi/4), -propWalls*math.pi/4, math.pi)
  c0=np.maximum(c0, c4)
  c5=linXY(pointData, posData, 0.1, -propBottom*math.pi/4, 0.0)
  c0=np.maximum(c0, c5)
  return c0


def combMin(pointData, posData, propWalls, propBottom):
  #propWalls...0.01-0.2
  c1=linXZ(pointData, posData, -1.0, propWalls*math.pi/4, 0.0)
  c0=c1
  c2=linYZ(pointData, posData, -1.0, propWalls*math.pi/4, 0.0)
  c0=np.minimum(c0, c2)
  c3=linXZ(pointData, posData, 1.0, propWalls*math.pi/4, math.pi)
  c0=np.minimum(c0, c3)
  c4=linYZ(pointData, posData, 1.0, propWalls*math.pi/4, math.pi)
  c0=np.minimum(c0, c4)
  c5=linXY(pointData, posData, 0.5, -propBottom*math.pi/4, 0.0)
  c0=np.minimum(c0, c5)
  return c0








# def combMax(pointData, posData, propWalls, propBottom):
  # #propWalls...0.2-0.4
  # c1=linXZ(pointData, posData, 0.0, -propWalls*math.pi/4, 0.0)
  # c0=c1
  # c2=linYZ(pointData, posData, 0.0, -propWalls*math.pi/4, 0.0)
  # c0=np.maximum(c0, c2)
  # c3=linXZ(pointData, posData, 0.0, propWalls*math.pi/4, 0.0)
  # c0=np.maximum(c0, c3)
  # c4=linYZ(pointData, posData, 0.0, propWalls*math.pi/4, 0.0)
  # c0=np.maximum(c0, c4)
  # c5=linXY(pointData, posData, 0.0, -propBottom*math.pi/4, 0.0)
  # c0=np.maximum(c0, c5)
  # return c0

# def combMin(pointData, posData, propWalls, propBottom):
  # #propWalls...0.01-0.2
  # c1=linXZ(pointData, posData, -1.0, propWalls*math.pi/4, 0.0)
  # c0=c1
  # c2=linYZ(pointData, posData, -1.0, propWalls*math.pi/4, 0.0)
  # c0=np.minimum(c0, c2)
  # c3=linXZ(pointData, posData, 1.0, -propWalls*math.pi/4, 0.0)
  # c0=np.minimum(c0, c3)
  # c4=linYZ(pointData, posData, 1.0, -propWalls*math.pi/4, 0.0)
  # c0=np.minimum(c0, c4)
  # c5=linXY(pointData, posData, 0.5, -propBottom*math.pi/4, 0.0)
  # c0=np.minimum(c0, c5)
  # return c0