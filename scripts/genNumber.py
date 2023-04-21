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

#python3 /system/user/mayr/git/bgnn/scripts/genNumber.py -problem hopper -experiment runs1 -saveModelName model -number 1

import os
import sys
import vtk
import vtk.util
import vtk.util.numpy_support
import numpy as np
import pickle
import argparse



#problem=sys.argv[1]
#experiment=sys.argv[2]
#saveModelName=sys.argv[3]

parser=argparse.ArgumentParser()
parser.add_argument("-problem", required=True, help="", type=str)
parser.add_argument("-experiment", required=True, help="", type=str)
parser.add_argument("-saveModelName", required=True, help="", type=str)
parser.add_argument("-number", help="", type=int, default=1)
args=parser.parse_args()

problem=args.problem
experiment=args.experiment
saveModelName=args.saveModelName



runDirBase="/system/user/mayr-data/BGNNRuns/"
modelDirBase=os.path.join(runDirBase, "models")
predDirBase=os.path.join(runDirBase, "predictions")
trajDirBase=os.path.join(runDirBase, "trajectories")

modelDir=os.path.join(modelDirBase, problem, experiment, saveModelName)
predDir=os.path.join(predDirBase, problem, experiment)
trajDir=os.path.join(trajDirBase, problem, experiment, saveModelName)



seqs=["_".join(x.split("_")[1:3]).split(".")[0] for x in os.listdir(trajDir) if x.startswith("predp_")]

for i in range(len(seqs)):
  saveFile=os.path.join(trajDir, "spgap_"+seqs[i]+".txt")
  file=open(saveFile, 'w')
  file.write(str(args.number))
  file.close()
