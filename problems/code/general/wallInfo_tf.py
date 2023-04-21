# Source codes of the publication "Boundary Graph Neural Networks for 3D Simulations"
# Distances between Triangles and Points in 3D on a GPU (Tensorflow 2 Code)
# Copyright (C) 2023 Andreas Mayr

#----------------------------------------------------------------------------

#The software here is based on software from David Eberly.
#
# David Eberly, Geometric Tools, Redmond WA 98052
# Copyright (c) 1998-2019
# Distributed under the Boost Software License, Version 1.0.
# http://www.boost.org/LICENSE_1_0.txt
# http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
# File Version: 3.0.0 (2016/06/19)
#
#
#ATTENTION:
#
#FOR THIS SOFTWARE, WE DID NOT TAKE ANY CARE FOR ANY POTENTIAL 
#NUMERICAL PROBLEMS (DATA TYPE CONVERSIONS, CUT-OFF ERRORS, 
#NUMERICAL PRECISION, ETC.). THIS SOFTWARE DOES VERY LIKELY 
#NOT EXACTLY REPRODUCE THE RESULTS FROM THE SOFTWARE OF 
#DAVID EBERLY. THIS SOFTWARE MAY BE BASED ON DIFFERENT DATA TYPES
#WITH DIFFERENT NUMERICAL PRECISIONS AND FOR FURTHER USAGE OF THIS
#SOFTWARE, ONE SHOULD PAY ATTENTION TO THIS.
#
#----------------------------------------------------------------------------
#
#Boost Software License - Version 1.0 - August 17th, 2003
#
#Permission is hereby granted, free of charge, to any person or organization
#obtaining a copy of the software and accompanying documentation covered by
#this license (the "Software") to use, reproduce, display, distribute,
#execute, and transmit the Software, and to prepare derivative works of the
#Software, and to permit third-parties to whom the Software is furnished to
#do so, all subject to the following:
#
#The copyright notices in the Software and this entire statement, including
#the above license grant, this restriction and the following disclaimer,
#must be included in all copies of the Software, in whole or in part, and
#all derivative works of the Software, unless such copies or derivative
#works are solely in the form of machine-executable object code generated by
#a source language processor.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
#SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
#FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
#ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#DEALINGS IN THE SOFTWARE.

#----------------------------------------------------------------------------

#As far as it is compatible with the Boost Software License - Version 1.0
#from above, the additional parts of this software, that extend the work of
#David Eberly are free software: you can redistribute them and/or modify
#them under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 (GPLv3) of the License .
#Otherwise (in case of incompatible License terms) the respective
#License terms from the Boost Software License - Version 1.0 
#have priority over the incompatible terms from GPLv3.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#The license file for GNU General Public License v3.0 is available here:
#https://github.com/ml-jku/bgnn/blob/master/licenses/own/LICENSE_GPL3
#----------------------------------------------------------------------------

#IN ANY CASE, THE FOLLOWING HOLDS FOR THIS SOFTWARE:

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
#SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
#FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
#ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#DEALINGS IN THE SOFTWARE.

#----------------------------------------------------------------------------


diff=tf.reshape(points,(-1,1,3))-tf.reshape(tcoord[:,0,:],(1,-1,3))
edge0=tcoord[:,1,:]-tcoord[:,0,:]
edge1=tcoord[:,2,:]-tcoord[:,0,:]
a00=tf.reshape(tf.reduce_sum(edge0*edge0, 1),(1,-1))
a01=tf.reshape(tf.reduce_sum(edge0*edge1, 1), (1,-1))
a11=tf.reshape(tf.reduce_sum(edge1*edge1, 1), (1,-1))
b0=-tf.reduce_sum(diff*tf.reshape(edge0, (1,-1,3)), 2)
b1=-tf.reduce_sum(diff*tf.reshape(edge1, (1,-1,3)), 2)
det=a00*a11-a01*a01
t0=a01*b1-a11*b0
t1=a01*b0-a00*b1

cond_1111=a11+b1-a01-b0>=a00-2.0*a01+a11
t0_1111=tf.where(cond_1111, 1.0, (a11+b1-a01-b0)/(a00-2.0*a01+a11))
t1_1111=tf.where(cond_1111, 0.0, 1.0-((a11+b1-a01-b0)/(a00-2.0*a01+a11)))

t0_1110=0.0
t1_1110=1.0

cond_111=a11+b1-a01-b0<=0.0
t0_111=tf.where(cond_111, t0_1110, t0_1111)
t1_111=tf.where(cond_111, t1_1110, t1_1111)

cond_11011=b0>=0.0
t0_11011=tf.where(cond_11011, 0.0, -b0/a00)
t1_11011=0.0

t0_11010=1.0
t1_11010=0.0

cond_1101=a00+b0<=0.0
t0_1101=tf.where(cond_1101, t0_11010, t0_11011)
t1_1101=tf.where(cond_1101, t1_11010, t1_11011)

#cond_1100=a11+b1-a01-b0>=a00-2.0*a01+a11
cond_1100=((a00+b0)-(a01+b1))>=(a00-(2.0)*a01+a11)
t0_1100=tf.where(cond_1100, 0.0, 1.0-(((a00+b0)-(a01+b1))/(a00-(2.0)*a01+a11)))
t1_1100=tf.where(cond_1100, 1.0, ((a00+b0)-(a01+b1))/(a00-(2.0)*a01+a11))

cond_110=a00+b0>a01+b1
t0_110=tf.where(cond_110, t0_1100, t0_1101)
t1_110=tf.where(cond_110, t1_1100, t1_1101)

cond_11=t1<0.0
t0_11=tf.where(cond_11, t0_110, t0_111)
t1_11=tf.where(cond_11, t1_110, t1_111)

cond_1011=b1>=0.0
t0_1011=0.0
t1_1011=tf.where(cond_1011, 0.0, -b1/a11)

t0_1010=0.0
t1_1010=1.0

cond_101=(a11+b1)<=0.0
t0_101=tf.where(cond_101, t0_1010, t0_1011)
t1_101=tf.where(cond_101, t1_1010, t1_1011)

cond_100=((a11+b1)-(a01+b0)) >= (a00-(2.0)*a01+a11)
t0_100=tf.where(cond_100, 1.0, ((a11+b1)-(a01+b0))/(a00-(2.0)*a01+a11))
t1_100=tf.where(cond_100, 0.0, 1.0-(((a11+b1)-(a01+b0))/(a00-(2.0)*a01+a11)))

cond_10=(a11+b1)>(a01+b0)
t0_10=tf.where(cond_10, t0_100, t0_101)
t1_10=tf.where(cond_10, t1_100, t1_101)

cond_1=t0<0.0
t0_1=tf.where(cond_1, t0_10, t0_11)
t1_1=tf.where(cond_1, t1_10, t1_11)

t0_011=(a01*b1-a11*b0)*(1.0/det)
t1_011=(a01*b0-a00*b1)*(1.0/det)

cond_0101=-b0>=a00
t0_0101=tf.where(cond_0101, 1.0, -b0/a00)
t1_0101=0.0

t0_0100=0.0
t1_0100=0.0

cond_010=b0 >= 0.0
t0_010=tf.where(cond_010, t0_0100, t0_0101)
t1_010=tf.where(cond_010, t1_0100, t1_0101)

cond_01=t1<0.0
t0_01=tf.where(cond_01, t0_010, t0_011)
t1_01=tf.where(cond_01, t1_010, t1_011)

cond_0011=-b1>=a11
t0_0011=0.0
t1_0011=tf.where(cond_0011, 1.0, -b1/a11)

t0_0010=0.0
t1_0010=0.0

cond_001=b1>=0.0
t0_001=tf.where(cond_001, t0_0010, t0_0011)
t1_001=tf.where(cond_001, t1_0010, t1_0011)

cond_00011=-b1>=a11
t0_00011=0.0
t1_00011=tf.where(cond_00011, 1.0, -b1/a11)

t0_00010=0.0
t1_00010=0.0

cond_0001=b1>=0.0
t0_0001=tf.where(cond_0001, t0_00010, t0_00011)
t1_0001=tf.where(cond_0001, t1_00010, t1_00011)

cond_0000=-b0>=a00
t0_0000=tf.where(cond_0000, 1.0, -b0/a00)
t1_0000=0.0

cond_000=b0<0.0
t0_000=tf.where(cond_000, t0_0000, t0_0001)
t1_000=tf.where(cond_000, t1_0000, t1_0001)

cond_00=t1<0.0
t0_00=tf.where(cond_00, t0_000, t0_001)
t1_00=tf.where(cond_00, t1_000, t1_001)

cond_0=t0<0.0
t0_0=tf.where(cond_0, t0_00, t0_01)
t1_0=tf.where(cond_0, t1_00, t1_01)

cond=t0+t1<=det
t0=tf.where(cond, t0_0, t0_1)
t1=tf.where(cond, t1_0, t1_1)

vecShape=(-1,)+tuple(edge0.shape)
factorShape=tuple(t0.shape)+(-1,)
startPoint=tf.reshape(tcoord[:,0,:], vecShape)
t0=tf.reshape(t0, factorShape)
t1=tf.reshape(t1, factorShape)
edge0=tf.reshape(edge0, vecShape)
edge1=tf.reshape(edge1, vecShape)
minPoint=startPoint+t0*edge0+t1*edge1

diffVec=tf.reshape(points, (-1,1,3))-minPoint
dists=tf.sqrt(tf.reduce_sum(diffVec**2, 2))


takeDist=tf.where(dists<neighborCutoff)
takeDist=tf.random.shuffle(takeDist)

#Version 1
# wallNodeData0=tf.concat([tf.zeros((minPoint.shape[1], nrPastVelocities*3))],1)
# wallNodeData1=tf.concat([tf.reduce_mean(minPoint, 0), tf.zeros((minPoint.shape[1], nrPastVelocities*3))],1)

# wallEdgeData_1=tf.concat([tf.gather_nd(diffVec, takeDist), tf.reshape(tf.gather_nd(dists, takeDist), (-1,1))], 1)
# wallEdgeData_2=tf.concat([-tf.gather_nd(diffVec, takeDist), tf.reshape(tf.gather_nd(dists, takeDist), (-1,1))], 1)
# wallEdgeData0=tf.concat([wallEdgeData_1, wallEdgeData_2], 0)

# wallEdgeData_1=tf.zeros((takeDist.shape[0],1))
# wallEdgeData_2=tf.zeros((takeDist.shape[0],1))
# wallEdgeData1=tf.concat([wallEdgeData_1, wallEdgeData_2], 0)

# srInd_1=takeDist[:,0].numpy()
# srInd_2=(takeDist[:,1]+xData.shape[1]).numpy()

# wallSenders=np.concatenate([srInd_1, srInd_2])
# wallReceivers=np.concatenate([srInd_2, srInd_1])


#Version 2

# minCoord=tf.gather_nd(minPoint, takeDist)
# minDist=tf.sqrt(tf.reduce_sum((minCoord-tf.reshape(minCoord,(-1,1,3)))**2,2))
# sameMat=(tf.reshape(takeDist[:,1], (-1,1))-tf.reshape(takeDist[:,1], (1,-1)))

# combineMat=tf.logical_and(minDist<neighborCutoff, sameMat==0)
# combineMat=tf.cast(combineMat, tf.float32)*(-1.)+tf.reshape(tf.range(0, 0.1, 0.1/combineMat.shape[0], dtype=tf.float32)[:combineMat.shape[0]], (1, -1))
# combineMap=tf.argmin(combineMat, 1)
# newComb=tf.gather(combineMap, combineMap)
# while not tf.reduce_all(newComb==combineMap):
  # combineMap=newComb
  # newComb=tf.gather(combineMap, combineMap)

# uInd=tf.unique(combineMap)
# wallNodeData0=tf.gather(minCoord, uInd[0])

# wallNodeData=tf.concat([wallNodeData0, tf.zeros((len(uInd[0]), nrPastVelocities*3))],1)
# particleWallData1=tf.concat([tf.gather_nd(diffVec, takeDist), tf.reshape(tf.gather_nd(dists, takeDist), (-1,1))], 1)
# particleWallData2=tf.concat([-tf.gather_nd(diffVec, takeDist), tf.reshape(tf.gather_nd(dists, takeDist), (-1,1))], 1)
# srInd1=takeDist[:,0].numpy()
# srInd2=(uInd[1]+xData.shape[1]).numpy()



# nodeFeatures=tf.concat([nodeFeatures, wallNodeData], 0)
# edgeFeatures=tf.concat([edgeFeatures, particleWallData], 0)
# senders=np.concatenate([senders, srInd1, srInd2])
# receivers=np.concatenate([receivers, srInd2, srInd1])

# nextPosition=tf.concat([nextPosition, wallNodeData[:,0:3]])
# currentPosition=tf.concat([currentPosition, wallNodeData[:,0:3]])
# currentVelocity=tf.concat([currentVelocity, tf.zeros((len(uInd[0]), 3))])







