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

rp["networkPar"]={
  "nrSteps": 3,
  "rese": False,
  "resn": False,
  "inputPar": {
    "nodep": {
      "actFun": "relu",
      "normeps": 1.0,
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": (rp["multNormV"]-1.0)+(rp["multParticle"]-1.0)+(rp["multWall"]-1.0)
    },
    "edgep": {
      "actFun": "relu",
      "normeps": 1.0,
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": rp["multAngle"]-1.0
    }
  },
  "processPar": {
    "sef": 0,
    "snf": 0,
    "useffeu": False,
    "usnffeu": False,
    "useffnu": False,
    "usnffnu": False,
    "rese": True,
    "resn": True,
    "nodep": {
      "normeps": 1.0,
      "lastAct": False,
      "layerN": True
    },
    "edgep": {
      "normeps": 1.0,
      "lastAct": False,
      "layerN": True
    }
  },
  "outputPar": {
    "nodep": {
      "normeps": 1.0,
      "actFun": "relu",
      "layerSz": [128, 3],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0.0
    },
    "edgep": {
      "normeps": 1.0,
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0.0
    }
  }
}
