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

cd /system/user/mayr-data/BGNN/drum/runs2
for sourceDir in {35..39}
do
for i in $(seq $((100+($sourceDir-35)*1)) $((100+($sourceDir-35+1)*1-1))); do  mkdir $i; cp $sourceDir/* $i; mkdir $i/meshes; cp $sourceDir/meshes/* $i/meshes/; mkdir $i/restart; cp $sourceDir/restart/* $i/restart/; mkdir $i/post; cp $sourceDir/post/init* $i/post/; done;
done

cd /system/user/mayr-data/BGNN/drum/runs2
for sourceDir in {100..104}
do
for i in $(seq $((105+($sourceDir-100)*5)) $((105+($sourceDir-100+1)*5-1))); do  mkdir $i; cp $sourceDir/* $i; mkdir $i/meshes; cp $sourceDir/meshes/* $i/meshes/; mkdir $i/restart; cp $sourceDir/restart/* $i/restart/; mkdir $i/post; cp $sourceDir/post/init* $i/post/; done;
done

for sourceDir in {100..129}
do
  cp s$sourceDir/myrseed.npy $sourceDir
done
