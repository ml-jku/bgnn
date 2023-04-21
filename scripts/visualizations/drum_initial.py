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



runid = '37'
gt=np.load("/system/user/mayr-data/BGNNRuns/trajectories/drum/runsi/model/gtp_"+runid+"_6.npy")
gt=gt[np.arange(0, gt.shape[0], 100)]
pred=np.load("/system/user/mayr-data/BGNNRuns/trajectories/drum/runsi/model/predp_"+runid+"_6.npy")
pred=pred[np.arange(0, pred.shape[0], 100)]


#(time, particle, coord) -> (particle, time, coord)
pred = pred.transpose((1,0,2))
gt = gt.transpose((1,0,2))



#flow
vpred = np.diff(pred,axis=1)
vgt = np.diff(gt,axis=1)



#get total min & max positions for binnings
minPred = np.min(pred,axis=(0,1))
maxPred = np.max(pred,axis=(0,1))
minGt = np.min(gt,axis=(0,1))
maxGt = np.max(gt,axis=(0,1))

minTot = np.minimum(minPred,minGt)
maxTot = np.maximum(maxPred,maxGt)

zbins = np.linspace(minTot[2],maxTot[2],20)
xbins = np.linspace(minTot[0],maxTot[0],20)



vpredlist = []
vgtlist = []
sum=0
#timeStep where averaging starts
tsStart = 30
for i in range(len(zbins)-1):
  mask = (pred[:,tsStart:,2]>=zbins[i]) * (pred[:,tsStart:,2]<zbins[i+1])
  mask = mask[:,:-1]
  vpredSel = vpred[:,tsStart:,:][mask]
  sum+=vpredSel.shape[0]
  VPredSel = vpredSel.sum(axis=0)
  vpredlist.append(VPredSel)
  
  mask = (gt[:,tsStart:,2]>=zbins[i]) * (gt[:,tsStart:,2]<zbins[i+1])
  mask = mask[:,:-1]
  vgtSel = vgt[:,tsStart:,:][mask]
  VGtSel = vgtSel.sum(axis=0)
  vgtlist.append(VGtSel)



zbinarr = np.array(zbins[:-1])

varrpred=np.array(vpredlist)
varrgt=np.array(vgtlist)




matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 20})
matplotlib.rc('axes', labelsize=30)
matplotlib.rc('legend', fontsize=30)
matplotlib.rc('text', usetex=True)
params= {'text.latex.preamble' : '\\usepackage{amsmath}\n'}
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

plt.plot(zbinarr,varrpred[:,0],'b--',label=r'$\text{Flow}\;x_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(zbinarr,varrgt[:,0],'b-',label=r'$\text{Flow}\;x_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)
plt.plot(zbinarr,varrpred[:,2],'r--',label=r'$\text{Flow}\;z_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(zbinarr,varrgt[:,2],'r-',label=r'$\text{Flow}\;z_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)

plt.xlabel(r"Position $z$", labelpad=17)

#rainbow_text is from code from https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib
#not included here

xrange=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]
yrange=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
xstart=plt.gca().get_xlim()[0]
ystart=plt.gca().get_ylim()[0]
#rainbow_text(xstart-xrange*0.05, ystart+yrange*0.5-250/2., ["Flow", "a", "x", "/", "/", "/", "z"], ["black", "white", "b", "white","black", "white", "r"],size=60)
legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(5, 5), linewidth=7.0, label='Prediction'), plt.Line2D([0], [0], color='black', linewidth=7.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fancybox=False, shadow=False, frameon=False, handletextpad=0.4, labelspacing=2, handlelength=4, columnspacing=1.)

plt.axes().yaxis.set_ticks(np.arange(-700, 200, 50))
plt.axes().xaxis.set_ticks(np.arange(-0.5, 0.2, 0.04))
plt.margins(0,0)
plt.tick_params(axis='both', which='major', pad=15)

plt.savefig(os.path.join(plotDir, "flowProfileZ_tsStart"+str(tsStart)+".pdf"))



ts = 30

x_max=0.5
x_min=-0.5
y_max=0.5
y_min=-0.5
z_max=0.
z_min=-0.5

xbins = np.linspace(x_min,x_max, num=10)
ybins = np.linspace(y_min,y_max, num=10)
zbins = np.linspace(z_min,z_max, num=10)

x_med=-0.05555556

ind_up = gt[:,ts,0]>=x_med
ind_down = gt[:,ts,0]<x_med

gt_up = gt[ind_up]
gt_down = gt[ind_down]

ind_up = pred[:,ts,0]>=x_med
ind_down = pred[:,ts,0]<x_med

pred_up = pred[ind_up]
pred_down = pred[ind_down]

harr_up_gt = []

for ts in range(gt_up.shape[1]):
  histo_up, _ = np.histogramdd(gt_up[:,ts,:], bins=[xbins,ybins,zbins])
  harr_up_gt.append(histo_up)

harr_down_gt = []

for ts in range(gt_down.shape[1]):
  histo_down, _ = np.histogramdd(gt_down[:,ts,:], bins=[xbins,ybins,zbins])
  harr_down_gt.append(histo_down)

harr_up_pred = []

for ts in range(pred_up.shape[1]):
  histo_up, _ = np.histogramdd(pred_up[:,ts,:], bins=[xbins,ybins,zbins])
  harr_up_pred.append(histo_up)

harr_down_pred = []

for ts in range(pred_down.shape[1]):
  histo_down, _ = np.histogramdd(pred_down[:,ts,:], bins=[xbins,ybins,zbins])
  harr_down_pred.append(histo_down)

def get_entropy_series(histos1, histos2):
  entr_ser = []
  for ts in range(len(histos1)):
    n_tot = np.sum(histos1[ts])+np.sum(histos2[ts])
    if n_tot == 0:
      entr_ser.append(0)
      continue
    entr = 0
    for hbin1,hbin2 in zip(histos1[ts].flatten(),histos2[ts].flatten()):
      if hbin1==0 or hbin2==0:
        continue
      fr1 = hbin1/(hbin1+hbin2)
      fr2 = hbin2/(hbin1+hbin2)
      entr += (hbin1+hbin2)/n_tot * -(fr1*np.log(fr1) + fr2*np.log(fr2))
    entr_ser.append(entr)
  return entr_ser

entr_ser_gt_x = get_entropy_series(harr_up_gt,harr_down_gt)
entr_ser_pred_x = get_entropy_series(harr_up_pred,harr_down_pred)



ts = 30

x_max=0.5
x_min=-0.5
y_max=0.5
y_min=-0.5
z_max=0.
z_min=-0.5

xbins = np.linspace(x_min,x_max, num=10)
ybins = np.linspace(y_min,y_max, num=10)
zbins = np.linspace(z_min,z_max, num=10)

z_med = -0.38888889

ind_up = gt[:,ts,2]>=z_med
ind_down = gt[:,ts,2]<z_med

gt_up = gt[ind_up]
gt_down = gt[ind_down]

ind_up = pred[:,ts,2]>=z_med
ind_down = pred[:,ts,2]<z_med

pred_up = pred[ind_up]
pred_down = pred[ind_down]

harr_up_gt = []

for ts in range(gt_up.shape[1]):
  histo_up, _ = np.histogramdd(gt_up[:,ts,:], bins=[xbins,ybins,zbins])
  harr_up_gt.append(histo_up)
  
harr_down_gt = []

for ts in range(gt_down.shape[1]):
  histo_down, _ = np.histogramdd(gt_down[:,ts,:], bins=[xbins,ybins,zbins])
  harr_down_gt.append(histo_down)

harr_up_pred = []

for ts in range(pred_up.shape[1]):
  histo_up, _ = np.histogramdd(pred_up[:,ts,:], bins=[xbins,ybins,zbins])
  harr_up_pred.append(histo_up)
  
harr_down_pred = []

for ts in range(pred_down.shape[1]):
  histo_down, _ = np.histogramdd(pred_down[:,ts,:], bins=[xbins,ybins,zbins])
  harr_down_pred.append(histo_down)

def get_entropy_series(histos1, histos2):
  entr_ser = []
  for ts in range(len(histos1)):
    n_tot = np.sum(histos1[ts])+np.sum(histos2[ts])
    if n_tot == 0:
      entr_ser.append(0)
      continue
    entr = 0
    for hbin1,hbin2 in zip(histos1[ts].flatten(),histos2[ts].flatten()):
      if hbin1==0 or hbin2==0:
        continue
      fr1 = hbin1/(hbin1+hbin2)
      fr2 = hbin2/(hbin1+hbin2)
      entr += (hbin1+hbin2)/n_tot * -(fr1*np.log(fr1) + fr2*np.log(fr2))
    entr_ser.append(entr)
  return entr_ser



entr_ser_gt_z = get_entropy_series(harr_up_gt,harr_down_gt)
entr_ser_pred_z = get_entropy_series(harr_up_pred,harr_down_pred)



matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 30})
matplotlib.rc('axes', labelsize=60)
matplotlib.rc('legend', fontsize=60)

plt.figure(figsize=(20,15))
plt.subplots_adjust(bottom=0.2, top=0.95) 
plt.grid(b=True, which='major', linewidth='1', linestyle='--', color='grey', dashes=(2,20,2,20))

plt.plot(np.arange(len(entr_ser_pred_x))[30:],entr_ser_pred_x[30:],'b--',label=r'$\text{Mixing Entropy}\;x_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(np.arange(len(entr_ser_gt_x))[30:],entr_ser_gt_x[30:],'b-',label=r'$\text{Mixing Entropy}\;x_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)

plt.plot(np.arange(len(entr_ser_pred_z))[30:],entr_ser_pred_z[30:],'r--',label=r'$\text{Mixing Entropy}\;z_{\scriptscriptstyle{\text{\;Prediction}}}$', dashes=(5, 5), linewidth=7.0)
plt.plot(np.arange(len(entr_ser_gt_z))[30:],entr_ser_gt_z[30:],'r-',label=r'$\text{Mixing Entropy}\;z_{\scriptscriptstyle{\text{\;Ground Truth}}}$', linewidth=7.0)


plt.xlabel("Time", labelpad=17)

#rainbow_text is from code from https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib
#not included here

xrange=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0]
yrange=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
xstart=plt.gca().get_xlim()[0]
ystart=plt.gca().get_ylim()[0]
#rainbow_text(xstart-xrange*0.05, yrange*0.1, ["Mixing Entropy (S)", "a", "x", "/", "/", "/", "z"], ["black", "white", "b", "white","black", "white", "r"],size=60)
legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(5, 5), linewidth=7.0, label='Prediction'), plt.Line2D([0], [0], color='black', linewidth=7.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fancybox=False, shadow=False, frameon=False, handletextpad=0.4, labelspacing=2, handlelength=4, columnspacing=1.)

plt.axes().xaxis.set_ticks(np.arange(0, 100, 10))
plt.margins(0,0)
plt.tick_params(axis='both', which='major', pad=15)

plt.savefig(os.path.join(plotDir, "entropy.pdf"))
plt.close()



