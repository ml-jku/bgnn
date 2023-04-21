#Source codes of the publication "Boundary Graph Neural Networks for 3D Simulations"
#Copyright (C) 2023  Sebastian Lehner, Andreas Mayr

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

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib import transforms
import warnings
warnings.filterwarnings("ignore")

homeDir=os.getenv('HOME')
plotDir=os.path.join(os.environ['HOME'], "bgnnPlots")



runid = '36'
gt=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runsi/model/gtp_"+runid+"_6.npy")
pred=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runsi/model/predp_"+runid+"_6.npy")
pred=pred[:gt.shape[0]]



#(time, particle, coord) -> (particle, time, coord)
pred = pred.transpose((1,0,2))
gt = gt.transpose((1,0,2))



# Particle averaged positions
avgPred = np.mean(pred[:,:,:],axis=0)
avgGt = np.mean(gt[:,:,:],axis=0)



#flow
vpred = np.diff(pred,axis=1)
vgt = np.diff(gt,axis=1)
flowPred = np.sum(vpred[:,:,:],axis=0)
flowGt = np.sum(vgt[:,:,:],axis=0)



matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 20})
matplotlib.rc('axes', labelsize=30)
matplotlib.rc('legend', fontsize=30)
matplotlib.rc('text', usetex=True)
params= {'text.latex.preamble' : r'\usepackage{amsmath}'}
params["legend.labelspacing"] = 0.15
params["legend.columnspacing"] = 0.7
params["legend.borderpad"] = 0.2
params["legend.handletextpad"] = 0.3
plt.rcParams.update(params)

matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 30})

matplotlib.rc('axes', labelsize=60)
matplotlib.rc('legend', fontsize=60)

plt.figure(figsize=(20,15))
plt.subplots_adjust(bottom=0.2, top=0.95) 
plt.grid(b=True, which='major', linewidth='1', linestyle='--', color='grey', dashes=(2,20,2,20))

plt.plot(avgPred[:,0],'b--',label=r'${x}_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(avgGt[:,0],'b-',label=r'$x_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)

plt.plot(avgPred[:,1],'g--',label=r'$y_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(avgGt[:,1],'g-',label=r'$y_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)

plt.plot(avgPred[:,2],'r--',label=r'$z_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(avgGt[:,2],'r-',label=r'$z_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)

plt.axes().xaxis.set_ticks(np.arange(0.0, 10, 1.0))
plt.axes().yaxis.set_ticks(np.arange(-0.05, 0.05, 0.007))
plt.margins(0,0)
plt.tick_params(axis='both', which='major', pad=15)

plt.xlabel("Time", labelpad=17)

#rainbow_text is from code from https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib
#not included here

xrange=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]
yrange=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
xstart=plt.gca().get_xlim()[0]
ystart=plt.gca().get_ylim()[0]
#rainbow_text(xstart-xrange*0.125, ystart+yrange*0.5-0.035/2., ["Position", "a", "x", "/", "/", "/", "y", "/", "/", "/","z"], ["black", "white", "b", "white","black", "white", "g", "white","black", "white", "r"],size=60)

legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(5, 5), linewidth=7.0, label='Prediction'), plt.Line2D([0], [0], color='black', linewidth=7.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fancybox=False, shadow=False, frameon=False, handletextpad=0.4, labelspacing=2, handlelength=4, columnspacing=1.)

plt.savefig(os.path.join(plotDir, "avgPos_hop2.pdf"))
plt.close()







matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 30})
matplotlib.rc('axes', labelsize=60)
matplotlib.rc('legend', fontsize=60)

plt.figure(figsize=(20,15))
plt.subplots_adjust(bottom=0.2, top=0.95) 
plt.grid(b=True, which='major', linewidth='1', linestyle='--', color='grey', dashes=(2,20,2,20))

plt.plot(flowPred[:,0],'b--',label=r'$\text{Flow}\;x_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(flowGt[:,0],'b-',label=r'$\text{Flow}\;x_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)

plt.plot(flowPred[:,1],'g--',label=r'$\text{Flow}\;y_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(flowGt[:,1],'g-',label=r'$\text{Flow}\;y_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)

plt.plot(flowPred[:,2],'r--',label=r'$\text{Flow}\;z_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(flowGt[:,2],'r-',label=r'$\text{Flow}\;z_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)


plt.xlabel("Time", labelpad=17)

#rainbow_text is from code from https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib
#not included here

xrange=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]
yrange=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
xstart=plt.gca().get_xlim()[0]
ystart=plt.gca().get_ylim()[0]
#rainbow_text(xstart-xrange*0.05, ystart+yrange*0.5-10/2., ["Flow", "a", "x", "/", "/", "/", "y", "/", "/", "/","z"], ["black", "white", "b", "white","black", "white", "g", "white","black", "white", "r"],size=60)

legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(5, 5), linewidth=7.0, label='Prediction'), plt.Line2D([0], [0], color='black', linewidth=7.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fancybox=False, shadow=False, frameon=False, handletextpad=0.4, labelspacing=2, handlelength=4, columnspacing=1.)

plt.axes().yaxis.set_ticks(np.arange(-20, 10, 2))
plt.margins(0,0)
plt.tick_params(axis='both', which='major', pad=15)

plt.savefig(os.path.join(plotDir, "flow_hop.pdf"))
plt.close()



