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

#def writeState(fileName, xlo, xhi, ylo, yhi, zlo, zhi, idData, xData, vData, omegaData, radius=2e-3, ptype=1, density=2500., atypes=2):

def writeState(fileName, xlo, xhi, ylo, yhi, zlo, zhi, idData, xData, vData, omegaData, radius, ptype, density, atypes):
  curTimeStep=0
  nrAtoms=xData.shape[0]
  
  if type(radius)!=np.ndarray:
    radius=np.repeat(radius, len(idData))
  if type(ptype)!=np.ndarray:
    ptype=np.repeat(ptype, len(idData))
  if type(density)!=np.ndarray:
    density=np.repeat(density, len(idData))
  
  
  
  genHeader="LAMMPS data file via write_data, version Version LIGGGHTS-PUBLIC 3.8.0, compiled 2019-03-01-16:14:25 by mayr, git commit 0242f0ba5adb6a0534d67bf9dd474966ff3a4775, timestep = {0}\n"+\
  "\n"+\
  "{1} atoms\n"+\
  "{2} atom types\n"+\
  "{3} {4} xlo xhi\n"+\
  "{5} {6} ylo yhi\n"+\
  "{7} {8} zlo zhi\n"+\
  "\n"+\
  "Atoms\n"+\
  "\n"
  genHeader=genHeader.format(curTimeStep, nrAtoms, atypes, xlo, xhi, ylo, yhi, zlo, zhi)
  #print(genHeader)

  velHeader="\n"+\
  "Velocities\n"+\
  "\n"

  ordLoc=np.array(["id", "type", "diameter", "density", "x", "y", "z", "im1", "im2", "im3"])

  myLoc=pd.DataFrame({
  "id": idData,
  "type": ptype,
  "diameter": 2.0*radius,
  "density": density,
  "x": xData[:,0],
  "y": xData[:,1],
  "z": xData[:,2],
  "im1": np.repeat(0, len(idData)),
  "im2": np.repeat(0, len(idData)),
  "im3": np.repeat(0, len(idData)),
  }).loc[:, ordLoc]


  ordVel=np.array(["id", "vx", "vy", "vz", "wx", "wy", "wz"])

  myVel=pd.DataFrame({
  "id": idData,
  "vx": vData[:,0],
  "vy": vData[:,1],
  "vz": vData[:,2],
  "wx": omegaData[:,0],
  "wy": omegaData[:,1],
  "wz": omegaData[:,2],
  }).loc[:, ordVel]


  dataFileText=genHeader+myLoc.to_csv(header=False, index=False, sep=" ")+velHeader+myVel.to_csv(header=False, index=False, sep=" ")

  file=open(fileName, "w")
  file.write(dataFileText)
  file.close()