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
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

homeDir=os.getenv('HOME')
plotDir=os.path.join(os.environ['HOME'], "bgnnPlots")



runid = '103'
gt=np.load("/system/user/mayr-data/BGNNRuns/trajectories/drum/runs1/model/gtp_"+runid+"_6.npy")
pred=np.load("/system/user/mayr-data/BGNNRuns/trajectories/drum/runs1/model/predp_"+runid+"_6.npy")

runs_gt = []

run_ids = list(range(105,120))


time1=50
time2=150

ind=time1 #use shift of 50
runs_gt.append(gt[ind:(gt.shape[0])])

ind=time2 #use shift of 150
runs_gt.append(gt[ind:(gt.shape[0])])

#particle number variation with same distributions
#10%: 105+(int(runid)-100)*5+0
#20%: 105+(int(runid)-100)*5+1
#30%: 105+(int(runid)-100)*5+2
#40%: 105+(int(runid)-100)*5+3
#50%: 105+(int(runid)-100)*5+4

accessId=105+(int(runid)-100)*5+4

ind=time1
runs_gt.append(np.load("/system/user/mayr-data/BGNNRuns/trajectories/drum/runs1/model/gtp_"+str(accessId)+"_6.npy")[ind:(gt.shape[0])])

ind=time2
runs_gt.append(np.load("/system/user/mayr-data/BGNNRuns/trajectories/drum/runs1/model/gtp_"+str(accessId)+"_6.npy")[ind:(gt.shape[0])])

#cut to have the same length for all
startInd=time2
endInd=gt.shape[0]-startInd
gt=gt[:endInd]
pred=pred[:endInd]
for i in range(0,len(runs_gt)):
  runs_gt[i]=runs_gt[i][0:endInd]

#main line added to uncertainty computation
runs_gt.append(gt)



#(time, particle, coord) -> (particle, time, coord)
pred = pred.transpose((1,0,2))
gt = gt.transpose((1,0,2))
for n in range(len(runs_gt)):
  runs_gt[n] = runs_gt[n].transpose((1,0,2))



#get total min & max positions for binnings
#use same binning as for non-cohesive (doing the computation here would give almost the same values, but for reproducibilty reasons of the plot take other version values)
minTot = np.array([-0.2401639 , -0.24169378, -0.24040242], dtype=np.float32)
maxTot = np.array([0.24034438, 0.24066177, 0.23996867], dtype=np.float32)

xbins = np.linspace(minTot[0],maxTot[0],20)
zbins = np.linspace(minTot[2],maxTot[2],20)

#input is array: part,time,dim
def get_zarr(pos):
  vlist=[]
  sum=0
  tsStart = 30
  v = np.diff(pos,axis=1)
  
  for i in range(len(zbins)-1):
    mask = (pos[:,tsStart:,2]>=zbins[i]) * (pos[:,tsStart:,2]<zbins[i+1])
    mask = mask[:,:-1]
    vSel = v[:,tsStart:,:][mask]
    sum+=vSel.shape[0]
    VSel = vSel.sum(axis=0)
    vlist.append(VSel)
  return np.array(vlist)



zbinarr = np.array(zbins[:-1])

vpred = get_zarr(pred)
vgt = get_zarr(gt)
vgt_run_list = []
for gtrun in runs_gt:
    vgt_run_list.append(get_zarr(gtrun))
rms_v_runs = np.std(np.array(vgt_run_list),axis=0)



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

matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 70})
axlsize=130
matplotlib.rc('axes', labelsize=axlsize)
matplotlib.rc('legend', fontsize=100)

plt.figure(figsize=(20,20))
plt.subplots_adjust(bottom=0.20, top=0.95, left=0.25, right=0.95) 
plt.grid(b=True, which='major', linewidth='1', linestyle='--', color='grey', dashes=(2,40,2,40))

plt.plot(zbinarr,vpred[:,0],'b--', dashes=(4, 4), linewidth=7.0)
plt.plot(zbinarr,vgt[:,0],'b-', linewidth=5.0)
plt.fill_between(zbinarr,vgt[:,0] - rms_v_runs[:,0], vgt[:,0] + rms_v_runs[:,0], color='b', alpha=0.2, linewidth=0.0)

plt.plot(zbinarr,vpred[:,2],'--',c="r", dashes=(4, 4), linewidth=7.0)
plt.plot(zbinarr,vgt[:,2],'-',c="r", linewidth=5.0)
plt.fill_between(zbinarr,vgt[:,2] - rms_v_runs[:,2], vgt[:,2] + rms_v_runs[:,2], color='r', alpha=0.2, linewidth=0.0)

inarry=np.concatenate([vpred[:,0], vpred[:,2], vgt[:,0], vgt[:,2]])


inarr=zbinarr
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().xaxis.set_ticks(lsp)
myarr=[inarr.min()-0.0*inarr.ptp(), inarr.max()+0.0*inarr.ptp()]
plt.xlim([myarr[0], myarr[1]])

inarr=inarry
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))*10
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))*10
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().yaxis.set_ticks(lsp)
myarr=[inarr.min()-0.1*inarr.ptp(), inarr.max()+0.1*inarr.ptp()]
plt.ylim([myarr[0]-0.35*(myarr[1]-myarr[0]), myarr[1]])

plt.tick_params(axis='both', which='major', pad=15)

ybox6=TextArea(r".\hspace*{50cm}\phantom{Flow $x$/}$z$\hspace*{50cm}.", textprops=dict(color="r",size=axlsize,rotation=90,ha='left',va='center'))
ybox5=TextArea(r".\hspace*{50cm}\phantom{Flow $x$}/\phantom{$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))
ybox2=TextArea(r".\hspace*{50cm}\phantom{Flow }$x$\phantom{/$z$}\hspace*{50cm}.", textprops=dict(color="b",size=axlsize,rotation=90,ha='left',va='center'))
ybox1=TextArea(r".\hspace*{50cm}Flow \phantom{$x$/$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))

xbox1=TextArea(r"Position z", textprops=dict(color="black", size=axlsize, ha='center', va='top'))

anchored_ybox6=AnchoredOffsetbox(loc='lower left', child=ybox6, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox5=AnchoredOffsetbox(loc='lower left', child=ybox5, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox2=AnchoredOffsetbox(loc='lower left', child=ybox2, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox1=AnchoredOffsetbox(loc='lower left', child=ybox1, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)

anchored_xbox1=AnchoredOffsetbox(loc='lower left', child=xbox1, pad=0.0, frameon=False, bbox_to_anchor=(0.5, -0.15), bbox_transform=plt.axes().transAxes, borderpad=0.)

plt.axes().add_artist(anchored_ybox6)
plt.axes().add_artist(anchored_ybox5)
plt.axes().add_artist(anchored_ybox2)
plt.axes().add_artist(anchored_ybox1)

plt.axes().add_artist(anchored_xbox1)


legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(4, 6), linewidth=7.0, label='Prediction'),
                 plt.Line2D([0], [0], color='black',                                linewidth=5.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=1, fancybox=False, shadow=False, frameon=False, handletextpad=0.5, labelspacing=0.0, handlelength=2.25, columnspacing=10, bbox_transform=plt.axes().transAxes)

plt.savefig(os.path.join(plotDir, "coh_zFlow_drum_"+runid+".png"))
plt.savefig(os.path.join(plotDir, "coh_zFlow_drum_"+runid+".pdf"))
plt.close()
im = Image.open(os.path.join(plotDir, "coh_zFlow_drum_"+runid+".png"))
im




def get_entropy_histos(arr,dim=2):
  ts = 0
  
  x_max=0.25
  x_min=-0.25
  y_max=0.25
  y_min=-0.25
  z_max=0.25
  z_min=-0.25

  xbins = np.linspace(x_min,x_max, num=10)
  ybins = np.linspace(y_min,y_max, num=10)
  zbins = np.linspace(z_min,z_max, num=10)

  arr_bins=[xbins,ybins,zbins]
  bins = arr_bins[dim]
  
  med = np.median(arr[:,ts,dim])
  idx = (np.abs(bins - med)).argmin()
  med = bins[idx]
  
  ind_up = arr[:,ts,dim]>=med
  ind_down = arr[:,ts,dim]<med

  gt_up = arr[ind_up]
  gt_down = arr[ind_down]    
  
  harr_up_gt = []

  for ts in range(gt_up.shape[1]):
    histo_up, _ = np.histogramdd(gt_up[:,ts,:], bins=[xbins,ybins,zbins])
    harr_up_gt.append(histo_up)

  harr_down_gt = []

  for ts in range(gt_down.shape[1]):
    histo_down, _ = np.histogramdd(gt_down[:,ts,:], bins=[xbins,ybins,zbins])
    harr_down_gt.append(histo_down)

  return  harr_down_gt, harr_up_gt

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



arr_up,arr_down = get_entropy_histos(pred,dim=0)
entr_ser_pred_x = get_entropy_series(arr_up,arr_down)
arr_up,arr_down = get_entropy_histos(pred)
entr_ser_pred_z = get_entropy_series(arr_up,arr_down)
arr_up,arr_down = get_entropy_histos(gt,dim=0)
entr_ser_gt_x = get_entropy_series(arr_up,arr_down)
arr_up,arr_down = get_entropy_histos(gt)
entr_ser_gt_z = get_entropy_series(arr_up,arr_down)
entr_list_x=[]
entr_list_z=[]
for run in runs_gt:
  arr_up, arr_down = get_entropy_histos(run,dim=0)
  entr_list_x.append(get_entropy_series(arr_up,arr_down))
  arr_up, arr_down = get_entropy_histos(run)
  entr_list_z.append(get_entropy_series(arr_up,arr_down))
entr_rms_x= np.std(np.array(entr_list_x),axis=0)
entr_rms_z= np.std(np.array(entr_list_z),axis=0)



matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 70})
axlsize=130
matplotlib.rc('axes', labelsize=axlsize)
matplotlib.rc('legend', fontsize=100)

plt.figure(figsize=(20,20))
plt.subplots_adjust(bottom=0.20, top=0.95, left=0.25, right=0.95) 
plt.grid(b=True, which='major', linewidth='1', linestyle='--', color='grey', dashes=(2,40,2,40))

plt.plot(np.linspace(1,235600,len(entr_ser_gt_z[::30])),entr_ser_pred_z[::30],'r--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,235600,len(entr_ser_gt_z[::30])),entr_ser_gt_z[::30],'r-', linewidth=5.0)
plt.fill_between(np.linspace(1,235600,len(entr_ser_gt_z[::30])),entr_ser_gt_z[::30] - entr_rms_z[::30], entr_ser_gt_z[::30] + entr_rms_z[::30], color='r', alpha=0.2, linewidth=0.0)

plt.plot(np.linspace(1,235600,len(entr_ser_gt_z[::30])),entr_ser_pred_x[::30],'b--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,235600,len(entr_ser_gt_z[::30])),entr_ser_gt_x[::30],'b-', linewidth=5.0)
plt.fill_between(np.linspace(1,235600,len(entr_ser_gt_z[::30])),entr_ser_gt_x[::30] - entr_rms_x[::30], entr_ser_gt_x[::30] + entr_rms_x[::30], color='b', alpha=0.2, linewidth=0.0)

plt.axes().yaxis.set_ticks([0,0.10,0.20,0.30])



inarr=np.linspace(1,235600,len(entr_ser_gt_z[::30]))
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))*1
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))*1
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().xaxis.set_ticks(lsp)
myarr=[inarr.min()-0.0*inarr.ptp(), inarr.max()+0.0*inarr.ptp()]
plt.xlim([myarr[0], myarr[1]])

inarr=np.concatenate([entr_ser_pred_z[::30], entr_ser_gt_z[::30], entr_ser_pred_x[::30], entr_ser_gt_x[::30]])
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))*10
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))*10
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().yaxis.set_ticks(lsp)
myarr=[inarr.min()-0.1*inarr.ptp(), inarr.max()+0.1*inarr.ptp()]
plt.ylim([myarr[0]-0.35*(myarr[1]-myarr[0]), myarr[1]])



plt.tick_params(axis='both', which='major', pad=15)
plt.axes().ticklabel_format(style='sci',axis='x',scilimits=(0,0))

ybox6=TextArea(r".\hspace*{50cm}\phantom{Mixing S $x$/}$z$\hspace*{50cm}.", textprops=dict(color="r",size=axlsize,rotation=90,ha='left',va='center'))
ybox5=TextArea(r".\hspace*{50cm}\phantom{Mixing S $x$}/\phantom{$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))
ybox2=TextArea(r".\hspace*{50cm}\phantom{Mixing S }$x$\phantom{/$z$}\hspace*{50cm}.", textprops=dict(color="b",size=axlsize,rotation=90,ha='left',va='center'))
ybox1=TextArea(r".\hspace*{50cm}Mixing S \phantom{$x$/$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))

xbox1=TextArea(r"Time", textprops=dict(color="black", size=axlsize, ha='center', va='top'))

anchored_ybox6=AnchoredOffsetbox(loc='lower left', child=ybox6, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox5=AnchoredOffsetbox(loc='lower left', child=ybox5, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox2=AnchoredOffsetbox(loc='lower left', child=ybox2, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox1=AnchoredOffsetbox(loc='lower left', child=ybox1, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)

anchored_xbox1=AnchoredOffsetbox(loc='lower left', child=xbox1, pad=0.0, frameon=False, bbox_to_anchor=(0.5, -0.15), bbox_transform=plt.axes().transAxes, borderpad=0.)

plt.axes().add_artist(anchored_ybox6)
plt.axes().add_artist(anchored_ybox5)
plt.axes().add_artist(anchored_ybox2)
plt.axes().add_artist(anchored_ybox1)

plt.axes().add_artist(anchored_xbox1)

legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(4, 6), linewidth=7.0, label='Prediction'),
                 plt.Line2D([0], [0], color='black',                                linewidth=5.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=1, fancybox=False, shadow=False, frameon=False, handletextpad=0.5, labelspacing=0.0, handlelength=2.25, columnspacing=10, bbox_transform=plt.axes().transAxes)

plt.savefig(os.path.join(plotDir, "coh_entropy"+runid+".png"))
plt.savefig(os.path.join(plotDir, "coh_entropy"+runid+".pdf"))
plt.close()
im = Image.open(os.path.join(plotDir, "coh_entropy"+runid+".png"))
im


