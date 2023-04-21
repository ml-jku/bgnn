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

#LOADLOADLOADLOADLOAD
#saveDir="/system/user/mayr-data/BGNNRuns/models/"
#os.listdir(saveDir)
#saveFilePrefix=saveDir+"d4_1592436421653_18-06-2020_gorilla5"+"/"

checkpoint=tf.train.Checkpoint(module=myGN, optimizer=optimizer)
latest=tf.train.latest_checkpoint(os.path.join(saveFilePrefix, "tf2Model"))
checkpoint.restore(latest)



f=open(os.path.join(saveFilePrefix, "trainInfo.pckl"), "rb")
epochNr=pickle.load(f)
trainNr=pickle.load(f)
trainList=pickle.load(f)
f.close()

f=open(os.path.join(saveFilePrefix, "parInfo.pckl"), "rb")
runParameters=pickle.load(f)
f.close()

f=open(os.path.join(saveFilePrefix, "estat.pckl"), "rb")
myeStat=pickle.load(f)
f.close()
estat=Statistics()
estat.setState(myeStat)

f=open(os.path.join(saveFilePrefix, "bstat.pckl"), "rb")
mybStat=pickle.load(f)
f.close()
bstat=Statistics()
bstat.setState(mybStat)
