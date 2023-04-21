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

#don't execute in ptiypthon or ptpython ==> does not work
import os
import pathlib
import multiprocessing as mp
import numpy as np

simDir=os.path.join("/system/user/mayr-data/BGNN/drum/")
runDir=os.path.join(simDir, "runs4")

def execLiggghts(ind):
  print("test"+str(ind)+"\n")
  os.environ['PATH']="/system/user/mayr-data/software/mpich/bin:"+os.environ['PATH']
  os.environ['PATH']="/system/user/mayr-data/software/liggghts:"+os.environ['PATH']
  os.environ['LD_LIBRARY_PATH']=""
  os.environ['LD_LIBRARY_PATH']="/system/user/mayr-data/software/mpich/lib:"+os.environ['LD_LIBRARY_PATH']
  os.environ['LD_LIBRARY_PATH']="/system/user/mayr-data/software/vtk/lib64:"+os.environ['LD_LIBRARY_PATH']
  #destPath=os.path.join(os.environ['HOME'], "scratch", "drum2", str(ind))
  destPath=os.path.join(runDir, str(ind))
  os.chdir(destPath)
  os.system("liggghts -in mainScript")
  return(0)

nrSimRunsStart=0
nrSimRuns=40
nrSimRunsStop=nrSimRunsStart+nrSimRuns

pool=mp.Pool(processes=40)
mypool=pool.imap_unordered(execLiggghts, range(nrSimRunsStart, nrSimRunsStop))
allres=[myres for myres in mypool]
pool.close()
#export PATH="/system/user/mayr-data/software/mpich/bin:"$PATH
#export PATH="/system/user/mayr-data/software/liggghts:"$PATH
#export LD_LIBRARY_PATH="/system/user/mayr-data/software/mpich/lib:"$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH="/system/user/mayr-data/software/vtk/lib64:"$LD_LIBRARY_PATH



