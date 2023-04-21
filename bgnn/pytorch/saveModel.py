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

#SAVESAVESAVESAVESAVE
hostname=os.uname()[1]
hostname=socket.gethostname()
saveDir="/system/user/mayr-data/BGNNRuns/models/" if "saveDir" not in dir() else saveDir

if not "saveFilePrefix" in dir():
  contNr=max([int(x.split('_')[0].split('d')[1]) for x in [y for y in os.listdir(saveDir) if y.startswith('d')]+['d-1_']])+1
  saveFilePrefix=saveDir+"d"+str(contNr)+"_"+str(int(time.time()*1000.0))+"_"+datetime.date.today().strftime("%d-%m-%Y")+"_"+hostname+"_"+os.environ['TMUX_ID']+"_"+os.environ['CUDA_VISIBLE_DEVICES']+"/"
if os.path.exists(os.path.join(saveFilePrefix)):
  backupFilePrefix=saveDir+"b"+saveFilePrefix.split("/")[-2]+"/"
  os.rename(saveFilePrefix, backupFilePrefix)
if not os.path.exists(os.path.join(saveFilePrefix)):
  os.makedirs(saveFilePrefix)

torch.save(myGN.cpu().state_dict(), os.path.join(saveFilePrefix, 'model.pt'))
torch.save({"optimizer": optimizer.state_dict()}, os.path.join(saveFilePrefix, 'opt.pt'))
myGN.to(device)

f=open(os.path.join(saveFilePrefix, "trainInfo.pckl"), "wb")
pickle.dump(epochNr, f, -1)
pickle.dump(trainNr, f, -1)
pickle.dump(trainList, f, -1)
f.close()

f=open(os.path.join(saveFilePrefix, "parInfo.pckl"), "wb")
pickle.dump(rp, f, -1)
f.close()

f=open(os.path.join(saveFilePrefix, "estat.pckl"), "wb")
pickle.dump(estat.getState(), f, -1)
f.close()

f=open(os.path.join(saveFilePrefix, "bstat.pckl"), "wb")
pickle.dump(bstat.getState(), f, -1)
f.close()

if "backupFilePrefix" in dir():
  if os.path.exists(os.path.join(backupFilePrefix)):
    if os.path.exists(os.path.join(backupFilePrefix+"model.pt")):
      os.remove(backupFilePrefix+"parInfo.pckl")
      os.remove(backupFilePrefix+"estat.pckl")
      os.remove(backupFilePrefix+"bstat.pckl")
      os.remove(backupFilePrefix+"model.pt")
      os.remove(backupFilePrefix+"opt.pt")
      os.remove(backupFilePrefix+"trainInfo.pckl")
      os.removedirs(backupFilePrefix)
    else:
      os.removedirs(backupFilePrefix)