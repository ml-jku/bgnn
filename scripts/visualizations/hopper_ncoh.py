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



runid = '104'
gt=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs2/model/gtp_"+runid+"_6.npy")
pred=np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs2/model/predp_"+runid+"_6.npy")

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
runs_gt.append(np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs2/model/gtp_"+str(accessId)+"_6.npy")[ind:(gt.shape[0])])

ind=time2
runs_gt.append(np.load("/system/user/mayr-data/BGNNRuns/trajectories/hopper/runs2/model/gtp_"+str(accessId)+"_6.npy")[ind:(gt.shape[0])])

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



# Particle averaged positions
avgPred = np.mean(pred[:,:,:],axis=0)
avgGt = np.mean(gt[:,:,:],axis=0)
avgGts = []
for n in range(len(runs_gt)):
  avgGts.append(np.mean(runs_gt[n][:,:,:],axis=0))
rmsGts = np.array(avgGts).std(axis = 0)



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

plt.plot(np.linspace(1,94240,len(avgPred)),avgPred[:,0],'b--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,94240,len(avgPred)),avgGt[:,0],'b-', linewidth=5.0)
plt.fill_between(np.linspace(1,94240,len(avgGt)),avgGt[:,0] - rmsGts[:,0], avgGt[:,0] + rmsGts[:,0], color='b', alpha=0.2, linewidth=0.0)


plt.plot(np.linspace(1,94240,len(avgPred)),avgPred[:,1],'g--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,94240,len(avgPred)),avgGt[:,1],'g-', linewidth=5.0)
plt.fill_between(np.linspace(1,94240,len(avgGt)),avgGt[:,1] - rmsGts[:,1], avgGt[:,1] + rmsGts[:,1], color='g', alpha=0.2, linewidth=0.0)

plt.plot(np.linspace(1,94240,len(avgPred)),avgPred[:,2],'r--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,94240,len(avgPred)),avgGt[:,2],'r-', linewidth=5.0)
plt.fill_between(np.linspace(1,94240,len(avgGt)),avgGt[:,2] - rmsGts[:,2], avgGt[:,2] + rmsGts[:,2], color='r', alpha=0.2, linewidth=0.0)

inarr=np.linspace(1,94240,len(avgPred))
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))*10
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))*10
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().xaxis.set_ticks(lsp)
myarr=[inarr.min()-0.0*inarr.ptp(), inarr.max()+0.0*inarr.ptp()]
plt.xlim([myarr[0], myarr[1]])

inarr=np.concatenate([avgPred[:,0], avgGt[:,0], avgPred[:,1], avgGt[:,1], avgPred[:,2], avgGt[:,2]])
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))*10
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))*10
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().yaxis.set_ticks(lsp)
myarr=[inarr.min()-0.1*inarr.ptp(), inarr.max()+0.1*inarr.ptp()]
plt.ylim([myarr[0]-0.35*(myarr[1]-myarr[0]), myarr[1]])

plt.tick_params(axis='both', which='major', pad=15)

ybox6=TextArea(r".\hspace*{50cm}\phantom{Position $x$/$y$/}$z$\hspace*{50cm}.", textprops=dict(color="r",size=axlsize,rotation=90,ha='left',va='center'))
ybox5=TextArea(r".\hspace*{50cm}\phantom{Position $x$/$y$}/\phantom{$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))
ybox4=TextArea(r".\hspace*{50cm}\phantom{Position $x$/}$y$\phantom{/$z$}\hspace*{50cm}.", textprops=dict(color="g",size=axlsize,rotation=90,ha='left',va='center'))
ybox3=TextArea(r".\hspace*{50cm}\phantom{Position $x$}/\phantom{$y$/$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))
ybox2=TextArea(r".\hspace*{50cm}\phantom{Position }$x$\phantom{/$y$/$z$}\hspace*{50cm}.", textprops=dict(color="b",size=axlsize,rotation=90,ha='left',va='center'))
ybox1=TextArea(r".\hspace*{50cm}Position \phantom{$x$/$y$/$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))

xbox1=TextArea(r"Time", textprops=dict(color="black", size=axlsize, ha='center', va='top'))

anchored_ybox6=AnchoredOffsetbox(loc='lower left', child=ybox6, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox5=AnchoredOffsetbox(loc='lower left', child=ybox5, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox4=AnchoredOffsetbox(loc='lower left', child=ybox4, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox3=AnchoredOffsetbox(loc='lower left', child=ybox3, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox2=AnchoredOffsetbox(loc='lower left', child=ybox2, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox1=AnchoredOffsetbox(loc='lower left', child=ybox1, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)

anchored_xbox1=AnchoredOffsetbox(loc='lower left', child=xbox1, pad=0.0, frameon=False, bbox_to_anchor=(0.5, -0.15), bbox_transform=plt.axes().transAxes, borderpad=0.)

plt.axes().add_artist(anchored_ybox6)
plt.axes().add_artist(anchored_ybox5)
plt.axes().add_artist(anchored_ybox4)
plt.axes().add_artist(anchored_ybox3)
plt.axes().add_artist(anchored_ybox2)
plt.axes().add_artist(anchored_ybox1)

plt.axes().add_artist(anchored_xbox1)


legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(4, 6), linewidth=7.0, label='Prediction'), 
                 plt.Line2D([0], [0], color='black',                                linewidth=5.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=1, fancybox=False, shadow=False, frameon=False, handletextpad=0.5, labelspacing=0.0, handlelength=2.25, columnspacing=10, bbox_transform=plt.axes().transAxes)
plt.axes().ticklabel_format(style='sci',axis='x',scilimits=(0,0))

plt.savefig(os.path.join(plotDir, "noncoh_avgPos_hop2_"+runid+".png"))
plt.savefig(os.path.join(plotDir, "noncoh_avgPos_hop2_"+runid+".pdf"))
plt.close()
im = Image.open(os.path.join(plotDir, "noncoh_avgPos_hop2_"+runid+".png"))
im





#flow
avgPredVel = np.diff(avgPred,axis=0)
avgGtVel = np.diff(avgGt,axis=0)
avgVGts_np = np.diff(np.array(avgGts),axis=1)
rmsGts = np.std(avgVGts_np, axis=0)





matplotlib.rc('font',**{'family':'serif','serif':['Times'], 'size' : 70})
axlsize=130
matplotlib.rc('axes', labelsize=axlsize)
matplotlib.rc('legend', fontsize=100)

plt.figure(figsize=(20,20))
plt.subplots_adjust(bottom=0.20, top=0.95, left=0.25, right=0.95) 
plt.grid(b=True, which='major', linewidth='1', linestyle='--', color='grey', dashes=(2,40,2,40))

plt.plot(np.linspace(1,94240,len(avgPredVel[::20])),avgPredVel[::20,0],'b--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,94240,len(avgPredVel[::20])),avgGtVel[::20,0],'b-', linewidth=5.0)
plt.fill_between(np.linspace(1,94240,len(avgPredVel[::20])),avgGtVel[::20,0] - rmsGts[::20,0], avgGtVel[::20,0] + rmsGts[::20,0], color='b', alpha=0.2, linewidth=0.0)

plt.plot(np.linspace(1,94240,len(avgPredVel[::20])),avgPredVel[::20,1],'g--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,94240,len(avgPredVel[::20])),avgGtVel[::20,1],'g-', linewidth=5.0)
plt.fill_between(np.linspace(1,94240,len(avgPredVel[::20])),avgGtVel[::20,1] - rmsGts[::20,1], avgGtVel[::20,1] + rmsGts[::20,1], color='g', alpha=0.2, linewidth=0.0)

plt.plot(np.linspace(1,94240,len(avgPredVel[::20])),avgPredVel[::20,2],'r--', dashes=(4, 4), linewidth=7.0)
plt.plot(np.linspace(1,94240,len(avgPredVel[::20])),avgGtVel[::20,2],'r-', linewidth=5.0)
plt.fill_between(np.linspace(1,94240,len(avgPredVel[::20])),avgGtVel[::20,2] - rmsGts[::20,2], avgGtVel[::20,2] + rmsGts[::20,2], color='r', alpha=0.2, linewidth=0.0)



inarr=np.linspace(1,94240,len(avgPredVel[::20]))
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))*10
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))*10
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().xaxis.set_ticks(lsp)
myarr=[inarr.min()-0.0*inarr.ptp(), inarr.max()+0.0*inarr.ptp()]
plt.xlim([myarr[0], myarr[1]])

inarr=np.concatenate([avgPredVel[::20,0], avgGtVel[::20,0], avgPredVel[::20,1], avgGtVel[::20,1], avgPredVel[::20,2], avgGtVel[::20,2]])
myarr=[inarr.min()+0.1*inarr.ptp(), inarr.max()-0.1*inarr.ptp()]
nr1=10**(np.ceil(np.log10(np.abs(myarr[0]))))*10
nr2=10**(np.ceil(np.log10(np.abs(myarr[1]))))*10
lsp=np.linspace(np.round(myarr[0]/min(nr1,nr2), 1)*min(nr1,nr2), np.round(myarr[1]/min(nr1,nr2), 1)*min(nr1,nr2), 3)
plt.axes().yaxis.set_ticks(lsp)
myarr=[inarr.min()-0.1*inarr.ptp(), inarr.max()+0.1*inarr.ptp()]
plt.ylim([myarr[0]-0.35*(myarr[1]-myarr[0]), myarr[1]])



plt.tick_params(axis='both', which='major', pad=15)
plt.axes().ticklabel_format(style='sci',axis='x',scilimits=(0,0))
plt.axes().ticklabel_format(style='sci',axis='y',scilimits=(0,0))
plt.axes().get_yaxis().get_offset_text().set_position((-0.15,0))

ybox6=TextArea(r".\hspace*{50cm}\phantom{Flow $x$/$y$/}$z$\hspace*{50cm}.", textprops=dict(color="r",size=axlsize,rotation=90,ha='left',va='center'))
ybox5=TextArea(r".\hspace*{50cm}\phantom{Flow $x$/$y$}/\phantom{$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))
ybox4=TextArea(r".\hspace*{50cm}\phantom{Flow $x$/}$y$\phantom{/$z$}\hspace*{50cm}.", textprops=dict(color="g",size=axlsize,rotation=90,ha='left',va='center'))
ybox3=TextArea(r".\hspace*{50cm}\phantom{Flow $x$}/\phantom{$y$/$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))
ybox2=TextArea(r".\hspace*{50cm}\phantom{Flow }$x$\phantom{/$y$/$z$}\hspace*{50cm}.", textprops=dict(color="b",size=axlsize,rotation=90,ha='left',va='center'))
ybox1=TextArea(r".\hspace*{50cm}Flow \phantom{$x$/$y$/$z$}\hspace*{50cm}.", textprops=dict(color="black", size=axlsize,rotation=90,ha='left',va='center'))

xbox1=TextArea(r"Time", textprops=dict(color="black", size=axlsize, ha='center', va='top'))

anchored_ybox6=AnchoredOffsetbox(loc='lower left', child=ybox6, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox5=AnchoredOffsetbox(loc='lower left', child=ybox5, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox4=AnchoredOffsetbox(loc='lower left', child=ybox4, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox3=AnchoredOffsetbox(loc='lower left', child=ybox3, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox2=AnchoredOffsetbox(loc='lower left', child=ybox2, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)
anchored_ybox1=AnchoredOffsetbox(loc='lower left', child=ybox1, pad=0.0, frameon=False, bbox_to_anchor=(-0.30, 0.5), bbox_transform=plt.axes().transAxes, borderpad=0.)

anchored_xbox1=AnchoredOffsetbox(loc='lower left', child=xbox1, pad=0.0, frameon=False, bbox_to_anchor=(0.5, -0.15), bbox_transform=plt.axes().transAxes, borderpad=0.)

plt.axes().add_artist(anchored_ybox6)
plt.axes().add_artist(anchored_ybox5)
plt.axes().add_artist(anchored_ybox4)
plt.axes().add_artist(anchored_ybox3)
plt.axes().add_artist(anchored_ybox2)
plt.axes().add_artist(anchored_ybox1)

plt.axes().add_artist(anchored_xbox1)

legend_elements=[plt.Line2D([0], [0], color='black', linestyle='--', dashes=(4, 6), linewidth=7.0, label='Prediction'),
                 plt.Line2D([0], [0], color='black',                                linewidth=5.0, label='Ground Truth')]
plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=1, fancybox=False, shadow=False, frameon=False, handletextpad=0.5, labelspacing=0.0, handlelength=2.25, columnspacing=10, bbox_transform=plt.axes().transAxes)

plt.savefig(os.path.join(plotDir, "noncoh_Flow_hop2_"+runid+".png"))
plt.savefig(os.path.join(plotDir, "noncoh_Flow_hop2_"+runid+".pdf"))
plt.close()
im = Image.open(os.path.join(plotDir, "noncoh_Flow_hop2_"+runid+".png"))
im


