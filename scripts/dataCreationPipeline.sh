#Source codes of the publication "Boundary Graph Neural Networks for 3D Simulations"
#Copyright (C) 2023 Andreas Mayr

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

cp ./data/bgnn.zip /system/user/mayr-data/
unzip bgnn.zip

for myscript in `find /system/user/mayr/git/bgnn|grep createInit|sort`
do
  python3 $myscript
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep execInit|sort`
do
  echo $myscript
  python3 $myscript
  echo end1
  echo end2
  echo end3
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep createMainCut|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep copyUncertainty|sort`
do
  echo $myscript
  bash $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep createUncertaintyCut|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep execMain|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep extractParticles|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep extractWalls|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep particleStatisticsLen|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

for myscript in `find /system/user/mayr/git/bgnn|grep particleStatisticsVec|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput

#needs a GPU (look not to get a memory overflow, i.e. verify that it was running correctly):
for myscript in `find /system/user/mayr/git/bgnn|grep particleWallDistances|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput


for myscript in `find /system/user/mayr/git/bgnn|grep extractParameters|sort`
do
  echo $myscript
  python3 $myscript
  echo end
done 1>>/system/user/mayr/bgnnPipelineOutput  2>>/system/user/mayr/bgnnPipelineOutput