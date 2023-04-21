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

nrConstraints=xSceneDataList[0].shape[1]
nrUConstraints=np.unique(xSceneDataList[0][0].reshape((-1,3)), axis=0).shape[0]

nrNodeFeatures=0
nrEdgeFeatures=0


if rp["usePastVelocitiesVec"][0]:
  nrNodeFeatures=nrNodeFeatures+rp["nrPastVelocities"]*3

if rp["usePastVelocitiesLen"][0]:
  nrNodeFeatures=nrNodeFeatures+rp["nrPastVelocities"]*1

if rp["usePastVelocitiesLen2"][0]:
  nrNodeFeatures=nrNodeFeatures+rp["nrPastVelocities"]*1

if rp["useWallDistLen"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints

if rp["useWallDistVec"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints*3

if rp["useWallInvDistLen"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints

if rp["useWallInvDistVec"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints*3

if rp["useWallInvDist2Len"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints

if rp["useWallInvDist2Vec"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints*3

if rp["useWallDistLenClip"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints

if rp["useWallInvDistLenClip"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints

if rp["useWallInvDistLen2Clip"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints

if rp["useWallDistLenClipInv"][0]:
  nrNodeFeatures=nrNodeFeatures+nrConstraints

if rp["useTPDistLen"][0]:
  nrNodeFeatures=nrNodeFeatures+nrUConstraints

if rp["useTPDistVec"][0]:
  nrNodeFeatures=nrNodeFeatures+nrUConstraints*3

if rp["useTPInvDistLen"][0]:
  nrNodeFeatures=nrNodeFeatures+nrUConstraints

if rp["useTPInvDistVec"][0]:
  nrNodeFeatures=nrNodeFeatures+nrUConstraints*3

if rp["useTPInvDist2Len"][0]:
  nrNodeFeatures=nrNodeFeatures+nrUConstraints

if rp["useTPInvDist2Vec"][0]:
  nrNodeFeatures=nrNodeFeatures+nrUConstraints*3

if rp["useNormalVec"][0]:
  nrNodeFeatures=nrNodeFeatures+6

if rp["useOneHotPE"][0]: #particle encoding
  nrNodeFeatures=nrNodeFeatures+2

if rp["useDistLen"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useDistVec"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useInvDistLen"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useInvDistVec"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useInvDist2Len"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useInvDist2Vec"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useDistLenVMod"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useDistVecVMod"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useInvDistLenVMod"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useInvDistVecVMod"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useInvDist2LenVMod"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useInvDist2VecVMod"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useUnitDistVec"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useProjectedUnitDistLenSenders"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useProjectedUnitDistVecSenders"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useProjectedUnitDistLen2Senders"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useProjectedUnitDistLenReceivers"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useProjectedUnitDistVecReceivers"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useProjectedUnitDistLen2Receivers"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useProjectedPartDistLenSum"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useProjectedPartDistVecSum"][0]:
  nrEdgeFeatures=nrEdgeFeatures+3

if rp["useProjectedPartDistLen2Sum"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if rp["useAngle"][0]:
  nrEdgeFeatures=nrEdgeFeatures+1

if nrNodeFeatures<2:
  nrNodeFeatures=nrNodeFeatures+2

if nrEdgeFeatures<2:
  nrEdgeFeatures=nrEdgeFeatures+2

nrNodeFeatures=nrNodeFeatures+1
