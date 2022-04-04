# The following code plots results from applying FFAVES and ESFW to the synthetic data presented in
# the Entropy Sorting paper

### Set folder path ###
path = "/Users/aradley/Python_Folders/Enviroments/FFAVES_Test/FFAVES"
path_2 = "/Synthetic_Data/Objects_From_Paper/"
###

### Import packages ###
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import pandas as pd
import matplotlib.colors as c
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import umap
###

### Load objects ###
# The ground truth synthetic data.
Complete_Synthetic_Data = pd.read_csv(path+path_2+"Complete_Synthetic_Data.csv",header=0,index_col=0)
# The synthetic data after intentionally adding false negative drop outs.
Drop_Out_Synthetic_Data = pd.read_csv(path+path_2+"Drop_Out_Synthetic_Data.csv",header=0,index_col=0)
# Complete_Discretised_Data represent 100% accurate discretisation of the data. I.e. the ground truth discretisation.
Complete_Discretised_Data = np.asarray(copy.copy(Complete_Synthetic_Data))
Complete_Discretised_Data[Complete_Discretised_Data > 0] = 1
# Intentionally sub-optimal discretisation cut-offs that were selected during the creation of the synthetic data.
Dicretisation_Cutoffs = np.load(path+path_2+"Dicretisation_Cutoffs.npy")

# Discretise the data according to these sub-optimal thresholds.
Discretised_Data = np.asarray(copy.copy(Drop_Out_Synthetic_Data))
for i in np.arange(Dicretisation_Cutoffs.shape[0]):
    Feature = Discretised_Data[:,i]
    Feature[Feature < Dicretisation_Cutoffs[i]] = 0
    Feature[Feature >= Dicretisation_Cutoffs[i]] = 1
    Discretised_Data[:,i] = Feature

Track_Percentage_Imputation = np.load(path+path_2+"Track_Percentage_Imputation.npy")
Sort_Gains = np.load(path+path_2+"Sort_Gains.npy")
Sort_Weights = np.load(path+path_2+"Sort_Weights.npy")
ES_Matrices_Features_Used_Inds = np.load(path+path_2+"ES_Matrices_Features_Used_Inds.npy")
Cycle_Suggested_Imputations = np.load(path+path_2+"Cycle_Suggested_Imputations.npy")
Track_Imputation_Steps = np.load(path+path_2+"Track_Imputation_Steps.npy",allow_pickle=True)
Optimised_Thresholds = np.load(path+path_2+"Optimised_Thresholds.npy")
# Using the final cycle of FFAVES suggested imputations
Chosen_Cycle = -1
Imputed_Discretised_Data = copy.copy(Discretised_Data)
Imputed_Discretised_Data[(Cycle_Suggested_Imputations[0],Cycle_Suggested_Imputations[1])] = (Imputed_Discretised_Data[(Cycle_Suggested_Imputations[0],Cycle_Suggested_Imputations[1])]*-1) + 1


########## Sythetic data plots #############

### Introduce Synthetic Data 1 ###

plt.figure(figsize=(7,6))
plt.title("Ground Truth",fontsize=16)
plt.xlabel("\n\n" + str(Complete_Synthetic_Data.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Complete_Synthetic_Data.shape[0]) + " Cells" ,fontsize=14)
ax = plt.gca()
im = ax.imshow(np.log(Complete_Synthetic_Data+1),cmap="hot",interpolation='nearest', aspect='auto')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax).set_label(label="$log_(Gene\ Expression)$",size=14)

plt.figure(figsize=(7,6))
plt.title("30% Random Dropouts",fontsize=16)
plt.xlabel("\n\n" + str(Drop_Out_Synthetic_Data.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Drop_Out_Synthetic_Data.shape[0]) + " Cells" ,fontsize=14)
ax = plt.gca()
im = ax.imshow(np.log(Drop_Out_Synthetic_Data+1),cmap="hot",interpolation='nearest', aspect='auto')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax).set_label(label="$log_(Gene\ Expression)$",size=14)

plt.show()

### Visualise how well FFAVES imputes FNs/FPs of SD1 ###

# Visualise functional and indescriminate FPs/FNs.

Imputations = np.zeros(Discretised_Data.shape)
FPs = np.load(path+path_2+"FPs.npy")
Imputations[(FPs[0],FPs[1])] = Imputations[(FPs[0],FPs[1])] + 1
Use = np.where(np.isin(FPs[1],np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])))[0]
Imputations[(FPs[0][Use],FPs[1][Use])] = Imputations[(FPs[0][Use],FPs[1][Use])] + 1
Use = np.where(np.isin(FPs[0],np.arange(1000,1200)))[0]
Imputations[(FPs[0][Use],FPs[1][Use])] = Imputations[(FPs[0][Use],FPs[1][Use])] + 1
colors = {"black":0,"cyan":1,"white":2}
l_colors = sorted(colors, key=colors.get)
cMap = c.ListedColormap(l_colors)
yticklabels = np.linspace(0,Imputations.shape[0],21)
plt.figure(figsize=(7,6))
plt.scatter(0,0,c=l_colors[1],label="Functional False Positives",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[2],label="Indiscriminate False Positives",zorder=-1,marker="s",edgecolors="k")
plt.legend(loc = "lower right")
sns.heatmap(Imputations,cmap=l_colors, vmin=0, vmax=len(colors),cbar=False,yticklabels=False,xticklabels=False)
plt.xlabel(str(Discretised_Data.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Discretised_Data.shape[0]) + " Cells" ,fontsize=14)
plt.title("False Positives Introduced to SD1",fontsize=16)

Imputations = np.zeros(Discretised_Data.shape)
FNs = np.load(path+path_2+"FNs.npy")
Imputations[(FNs[0],FNs[1])] = Imputations[(FNs[0],FNs[1])] + 1
Use = np.where(np.isin(FNs[1],np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])))[0]
Imputations[(FNs[0][Use],FNs[1][Use])] = Imputations[(FNs[0][Use],FNs[1][Use])] + 1
Use = np.where(np.isin(FNs[0],np.arange(1000,1200)))[0]
Imputations[(FNs[0][Use],FNs[1][Use])] = Imputations[(FNs[0][Use],FNs[1][Use])] + 1
colors = {"black":0,"cyan":1,"white":2}
l_colors = sorted(colors, key=colors.get)
cMap = c.ListedColormap(l_colors)
yticklabels = np.linspace(0,Imputations.shape[0],21)
plt.figure(figsize=(7,6))
plt.scatter(0,0,c=l_colors[1],label="Functional False Negatives",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[2],label="Indiscriminate False Negatives",zorder=-1,marker="s",edgecolors="k")
plt.legend(loc = "lower right")
sns.heatmap(Imputations,cmap=l_colors, vmin=0, vmax=len(colors),cbar=False,yticklabels=False,xticklabels=False)
plt.xlabel(str(Discretised_Data.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Discretised_Data.shape[0]) + " Cells" ,fontsize=14)
plt.title("False Negatives Introduced to SD1",fontsize=16)

plt.show()

### Visualise how well FFAVES imputes the precision and recall of FFAVES FN/FP identification ###

Structured_Expression = np.load(path+path_2+"Structured_Expression.npy")
##

## FP Recall ##
Imputations = np.zeros(Discretised_Data.shape)
Imputations[(FPs[0],FPs[1])] = Imputations[(FPs[0],FPs[1])] + 1
Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] = 0
Imputations[np.arange(1000,1200),:] = 0
Num_Ground_Truth = np.sum(Imputations)

Imputations = copy.copy(Suggested_FPs)
Imputations[(FPs[0],FPs[1])] = Imputations[(FPs[0],FPs[1])] + 2
Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] = Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] * 5
Imputations[np.arange(1000,1200),:] = Imputations[np.arange(1000,1200),:] * 5
Ground_FPs_Identified = np.where(Imputations==3)
Num_Ground_FPs_Identified = Ground_FPs_Identified[0].shape[0]
Ground_FPs_Not_Identified = np.where(Imputations==2)
Num_Ground_FPs_Not_Identified = Ground_FPs_Not_Identified[0].shape[0]
Imputations[Imputations > 3] = 3

Identified = np.round(Num_Ground_FPs_Identified/Num_Ground_Truth,3) *100
Not_Identified = np.round(Num_Ground_FPs_Not_Identified/Num_Ground_Truth,3) *100
plt.figure(figsize=(3.5,6))
p1 = plt.bar(["Identified\n(Recall)","Not Identified"],[Identified,Not_Identified],color=['#1f77b4', '#ff7f0e'])
plt.ylabel("Percentage (%)",fontsize=13)
plt.ylim(0,100)
plt.title("Recall of Ground Truth \n FPs Identified",fontsize=14)
plt.bar_label(p1,fontsize=12)
plt.xticks(fontsize=13)
plt.tight_layout()

Imputations = np.zeros(Suggested_FPs.shape)
Imputations[(FPs[0],FPs[1])] = 1
Imputations[Ground_FPs_Identified] = 1
Imputations[Ground_FPs_Not_Identified] = 2
Imputations[np.arange(1000,1200)] = Imputations[np.arange(1000,1200),:] * 3
Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] = Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] * 3
Imputations[Imputations > 3] = 3
colors = {"black":0,"#1f77b4":1,"#ff7f0e":2,"red":3}
l_colors = sorted(colors, key=colors.get)
cMap = c.ListedColormap(l_colors)
yticklabels = np.linspace(0,Imputations.shape[0],21)
plt.figure(figsize=(7,6))
plt.scatter(0,0,c=l_colors[1],label="Sucessfully identified FPs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[2],label="Failed to identify FPs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[3],label="Ignored for recall calculation",zorder=-1,marker="s",edgecolors="k")
plt.legend(loc = "lower right")
sns.heatmap(Imputations,cmap=l_colors, vmin=0, vmax=len(colors),cbar=False,yticklabels=False,xticklabels=False)
plt.xlabel(str(Imputations.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Imputations.shape[0]) + " Cells" ,fontsize=14)
plt.title("Recall of Identified FPs",fontsize=16)

plt.show()

## FP Precision ##

Imputations = np.zeros(Discretised_Data.shape)
Imputations[(Structured_Expression[0],Structured_Expression[1])] = Imputations[(Structured_Expression[0],Structured_Expression[1])] + 1
Imputations[(FPs[0],FPs[1])] = Imputations[(FPs[0],FPs[1])] + 2
Imputations = Imputations + (Suggested_FPs*3)
Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] = Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] * 5
Imputations[np.arange(1000,1200),:] = Imputations[np.arange(1000,1200),:] * 5

Real_FPs = np.where(Imputations == 5)
Num_Real_FPs = Real_FPs[0].shape[0]
Fake_FPs = np.where(Imputations == 4)
Num_Fake_FPs = Fake_FPs[0].shape[0]
Num_FFAVES_Identified = np.sum(Imputations==5) + np.sum(Imputations==4)

Identified = np.round(Num_Real_FPs/Num_FFAVES_Identified,3) *100
Not_Identified = np.round(Num_Fake_FPs/Num_FFAVES_Identified,3) *100
plt.figure(figsize=(3.5,6))
p1 = plt.bar(["Correct FPs\n(Precision)","Incorrect FPs"],[Identified,Not_Identified],color=['#1f77b4', '#ff7f0e'])
plt.ylabel("Percentage (%)",fontsize=13)
plt.ylim(0,100)
plt.title("Precision of FPs\nIdentified by FFAVES",fontsize=14)
plt.bar_label(p1,fontsize=12)
plt.xticks(fontsize=13)
plt.tight_layout()

Imputations = np.zeros(Discretised_Data.shape)
Imputations = Imputations + (Suggested_FPs*3)
Imputations[Real_FPs] = 1
Imputations[Fake_FPs] = 2

colors = {"black":0,"#1f77b4":1,"#ff7f0e":2,"red":3}
l_colors = sorted(colors, key=colors.get)
cMap = c.ListedColormap(l_colors)
yticklabels = np.linspace(0,Imputations.shape[0],21)
plt.figure(figsize=(7,6))
plt.scatter(0,0,c=l_colors[1],label="Real ground truth FPs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[2],label="Inccorectly assigned as FPs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[3],label="Ignored for precision calculation",zorder=-1,marker="s",edgecolors="k")
plt.legend(loc = "lower right")
sns.heatmap(Imputations,cmap=l_colors, vmin=0, vmax=len(colors),cbar=False,yticklabels=False,xticklabels=False)
plt.xlabel(str(Imputations.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Imputations.shape[0]) + " Cells" ,fontsize=14)
plt.title("Precision of Identified FPs",fontsize=16)

plt.show()

## FN Recall ##
Imputations = copy.copy(Imputed_Discretised_Data)
Imputations[(Structured_Expression[0],Structured_Expression[1])] = Imputations[(Structured_Expression[0],Structured_Expression[1])] + 2
Imputations = Imputations + Discretised_Data
Imputations[(FPs[0],FPs[1])] = Imputations[(FPs[0],FPs[1])] + 5
Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] = 0
Imputations[np.arange(1000,1200),:] = 0
Imputations[Imputations>3] = 0
# 1 = Faslsy identified, 2 = Failed to identify, 3 = successfuly identified

Fake_FNs = np.where(Imputations==1)
Num_Fake_FNs = Fake_FNs[0].shape[0]
Missed_FNs = np.where(Imputations==2)
Num_Missed_FNs = Missed_FNs[0].shape[0]
Identified_FNs = np.where(Imputations==3)
Num_Identified_FNs = Identified_FNs[0].shape[0]
Num_Ground_Truth = Num_Identified_FNs + Num_Missed_FNs

Identified = np.round(Num_Identified_FNs/Num_Ground_Truth,3) *100
Not_Identified = np.round(Num_Missed_FNs/Num_Ground_Truth,3) *100
plt.figure(figsize=(3.5,6))
p1 = plt.bar(["Identified\n(Recall)","Not Identified"],[Identified,Not_Identified],color=['#1f77b4', '#ff7f0e'])
plt.ylabel("Percentage (%)",fontsize=13)
plt.ylim(0,100)
plt.title("Recall of Ground Truth\nFNs Identified",fontsize=14)
plt.bar_label(p1,fontsize=12)
plt.xticks(fontsize=13)
plt.tight_layout()

Imputations = np.zeros(Discretised_Data.shape)
Imputations[(FNs[0],FNs[1])] = 3
Imputations[Identified_FNs] = 1
Imputations[Missed_FNs] = 2
Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] = 0

colors = {"black":0,"#1f77b4":1,"#ff7f0e":2,"red":3}
l_colors = sorted(colors, key=colors.get)
cMap = c.ListedColormap(l_colors)
yticklabels = np.linspace(0,Imputations.shape[0],21)
plt.figure(figsize=(7,6))
plt.scatter(0,0,c=l_colors[1],label="Sucessfully identified FNs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[2],label="Failed to identify FNs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[3],label="Ignored for recall calculation",zorder=-1,marker="s",edgecolors="k")
plt.legend(loc = "lower right")
sns.heatmap(Imputations,cmap=l_colors, vmin=0, vmax=len(colors),cbar=False,yticklabels=False,xticklabels=False)
plt.xlabel(str(Imputations.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Imputations.shape[0]) + " Cells" ,fontsize=14)
plt.title("Recall of Identified FNs",fontsize=16)

plt.show()

## FN Precision ##
Imputations = copy.copy(Suggested_FNs)
Imputations[(Structured_Expression[0],Structured_Expression[1])] = Imputations[(Structured_Expression[0],Structured_Expression[1])] + 2
Imputations[:,np.arange(Discretised_Data.shape[1]-550,Discretised_Data.shape[1])] = 0
Imputations[np.arange(1000,1200),:] = 0

Real_FNs = np.where(Imputations==3)
Num_Real_FNs = Real_FNs[0].shape[0]
Fake_FNs = np.where(Imputations==1)
Num_Fake_FNs = Fake_FNs[0].shape[0]
Num_FFAVES_Identified = np.sum(Imputations==3) + np.sum(Imputations==1)

Incorrect_FNs = np.round(Num_Fake_FNs/Num_FFAVES_Identified,3) *100
Correct_FNs = np.round(Num_Real_FNs/Num_FFAVES_Identified,3) *100
plt.figure(figsize=(3.5,6))
p1 = plt.bar(["Correct FNs\n(Precision)","Incorrect FNs"],[Correct_FNs,Incorrect_FNs],color=['#1f77b4', '#ff7f0e'])
plt.ylabel("Percentage (%)",fontsize=13)
plt.ylim(0,100)
plt.title("Precision of FNs\nIdentified by FFAVES",fontsize=14)
plt.bar_label(p1,fontsize=12)
plt.xticks(fontsize=13)
plt.tight_layout()

Imputations = copy.copy(Suggested_FNs)*3
Imputations[Real_FNs] = 1
Imputations[Fake_FNs] = 2

colors = {"black":0,"#1f77b4":1,"#ff7f0e":2,"red":3}
l_colors = sorted(colors, key=colors.get)
cMap = c.ListedColormap(l_colors)
yticklabels = np.linspace(0,Imputations.shape[0],21)
plt.figure(figsize=(7,6))
plt.scatter(0,0,c=l_colors[1],label="Real ground truth FNs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[2],label="Inccorectly assigned as FNs",zorder=-1,marker="s",edgecolors="k")
plt.scatter(0,0,c=l_colors[3],label="Ignored for precision calculation",zorder=-1,marker="s",edgecolors="k")
plt.legend(loc = "lower right")
sns.heatmap(Imputations,cmap=l_colors, vmin=0, vmax=len(colors),cbar=False,yticklabels=False,xticklabels=False)
plt.xlabel(str(Imputations.shape[1]) + " Genes",fontsize=14)
plt.ylabel(str(Imputations.shape[0]) + " Cells" ,fontsize=14)
plt.title("Precision of Identified FNs",fontsize=16)

plt.show()


#### Visualise Threshold Optimisation ####

Initial_Errors = np.zeros(Dicretisation_Cutoffs.shape[0])
Optimised_Errors = copy.copy(Initial_Errors)
for i in np.arange(Dicretisation_Cutoffs.shape[0]):
    Feature = np.asarray(Complete_Synthetic_Data)[:,i]
    # Sub optimial discretisation error
    Initial_Error = np.where(np.logical_and(Feature > 0,Feature <= Dicretisation_Cutoffs[i]))[0].shape[0]
    Initial_Errors[i] = Initial_Error
    if Optimised_Thresholds[i] != -1:
        # Optimised error
        Optimised_Error = np.where(np.logical_and(Feature > 0,Feature <= Optimised_Thresholds[i]))[0].shape[0]
        Optimised_Errors[i] = Optimised_Error
    else:
        Optimised_Errors[i] = Initial_Error


Imputations = np.zeros(Discretised_Data.shape)
Imputations[Missed_FNs] = 1
Imputations[Identified_FNs] = 2

Errors_Found = np.sum(Imputations==2,axis=0)[np.arange(0,969)]
Change = (Optimised_Errors/Initial_Errors)[np.arange(0,969)]
No_Optimisation = np.where(Optimised_Thresholds[np.arange(0,969)]==-1)[0]
Is_Optimisation = np.where(Optimised_Thresholds[np.arange(0,969)]!=-1)[0]
Total_Error = np.sum(np.asarray(Imputations!=0),axis=0)[np.arange(0,969)]
Ratios = Errors_Found/Total_Error

fig, ax = plt.subplots()
scat2 = ax.scatter(No_Optimisation,Change[No_Optimisation],c="r",s=10,label="No optimisation perfomed",marker="s")
ax.scatter(0,0,s=8,label="Optimisation perfomed",c="k",zorder=-1)
scat = ax.scatter(Is_Optimisation,Change[Is_Optimisation],c=Ratios[Is_Optimisation],s=8,cmap="magma",vmin=0,vmax=1)
plt.ylabel("Percentage change in error\nafter optimisation (%)",fontsize=13)
plt.xlabel("Gene ID",fontsize=13)
plt.title("Ground Truth Error\nOriginal Vs Optimised Thresholds",fontsize=15)
cb = plt.colorbar(scat)
cb.set_label(label="Fraction of ground truth\nerror identified by FFAVES", fontsize=14)
plt.axhline(1,linestyle="--",c="g",label="No change after optimisation",zorder=-1,alpha=0.5)
plt.legend()
ax.set_yticks(np.arange(0,2.5,0.5))
ax.set_yticklabels(np.arange(0,2.5,0.5)*100)
#plt.annotate("92",(92+5,Change[92]+0.05),backgroundcolor='w')
#plt.annotate("261",(261+5,Change[261]+0.05),backgroundcolor='w')
#plt.annotate("231",(231+5,Change[231]+0.05),backgroundcolor='w')
#plt.annotate("385",(385+5,Change[385]+0.05),backgroundcolor='w')

plt.show()

# Best example
Ind = 275

plt.figure()
plt.hist(Complete_Synthetic_Data[str(Ind)],bins=30)
plt.axvline(Dicretisation_Cutoffs[Ind],c="r",label="Initial Discretisation Threshold")
plt.axvline(Optimised_Thresholds[Ind],c="g",label="Optimised Discretisation Threshold")
plt.legend(facecolor="white",framealpha=1)
plt.title("Gene ID: " + str(Ind),fontsize=15)
plt.ylabel("Frequency",fontsize=13)
plt.xlabel("Gene Expression",fontsize=13)

# Wrong way example
Ind = int(np.where(Change==np.max(Change))[0][0])

plt.figure()
plt.hist(Complete_Synthetic_Data[str(Ind)],bins=30)
plt.axvline(Dicretisation_Cutoffs[Ind],c="r",label="Initial Discretisation Threshold")
plt.axvline(Optimised_Thresholds[Ind],c="g",label="Optimised Discretisation Threshold")
plt.legend(facecolor="white",framealpha=1)
plt.title("Gene ID: " + str(Ind),fontsize=15)
plt.ylabel("Frequency",fontsize=13)
plt.xlabel("Gene Expression",fontsize=13)

# No change example
Ind = 595

plt.figure()
plt.hist(Complete_Synthetic_Data[str(Ind)],bins=30)
plt.axvline(Dicretisation_Cutoffs[Ind],c="r",label="Initial Discretisation Threshold")
plt.axvline(Optimised_Thresholds[Ind],c="g",label="Optimised Discretisation Threshold")
plt.legend(facecolor="white",framealpha=1)
plt.title("Gene ID: " + str(Ind),fontsize=15)
plt.ylabel("Frequency",fontsize=13)
plt.xlabel("Gene Expression",fontsize=13)

# Small change example
Ind = 365

plt.figure()
plt.hist(Complete_Synthetic_Data[str(Ind)],bins=30)
plt.axvline(Dicretisation_Cutoffs[Ind],c="r",label="Initial Discretisation Threshold")
plt.axvline(Optimised_Thresholds[Ind],c="g",label="Optimised Discretisation Threshold")
plt.legend(facecolor="white",framealpha=1)
plt.title("Gene ID: " + str(Ind),fontsize=15)
plt.ylabel("Frequency",fontsize=13)
plt.xlabel("Gene Expression",fontsize=13)

plt.show()



### Create Gene Clustering Plots, including Silhoutte scores ######
Dropout_Sort_Gains = np.load(path+path_2+"Dropout_Sort_Gains.npy")
Dropout_Sort_Weights = np.load(path+path_2+"Dropout_Sort_Weights.npy")
FFAVES_Imputed_Sort_Gains = np.load(path+path_2+"FFAVES_Imputed_Sort_Gains.npy")
FFAVES_Imputed_Sort_Weights = np.load(path+path_2+"FFAVES_Imputed_Sort_Weights.npy")
Ground_Truth_Sort_Gains = np.load(path+path_2+"Ground_Truth_Sort_Gains.npy")
Ground_Truth_Sort_Weights = np.load(path+path_2+"Ground_Truth_Sort_Weights.npy")

# Gene groups: 0-49,50-247,248-431,432-612,613-797,798-968,967-1017,1018-1518
#Gene_Groups = np.array([[0,49],[50,247],[248,431],[432,612],[613,797],[798,968],[967,1017],[1018,1518]])
Gene_Groups = np.array([[0,49],[50,247],[248,431],[432,612],[613,797],[798,968],[967,1518]])
Gene_Labels = np.zeros(Sort_Gains.shape[0])
for i in np.arange(Gene_Groups.shape[0]):
    Gene_Labels[np.arange(Gene_Groups[i][0],Gene_Groups[i][1]+1)] = i

IG_silhouette_vals_0 = silhouette_samples(Dropout_Sort_Gains,Gene_Labels)
IG_silhouette_vals_1 = silhouette_samples(FFAVES_Imputed_Sort_Gains,Gene_Labels)
ESS_silhouette_vals_0 = silhouette_samples(Dropout_Sort_Gains*Dropout_Sort_Weights,Gene_Labels)
ESS_silhouette_vals_1 = silhouette_samples(FFAVES_Imputed_Sort_Gains*FFAVES_Imputed_Sort_Weights,Gene_Labels)
GT_IG_silhouette_vals= silhouette_samples(Ground_Truth_Sort_Gains,Gene_Labels)
GT_ESS_silhouette_vals = silhouette_samples(Ground_Truth_Sort_Gains*Ground_Truth_Sort_Weights,Gene_Labels)

imshow_x_lim_min = np.min(np.array([np.min(Dropout_Sort_Gains*Dropout_Sort_Weights),np.min(Ground_Truth_Sort_Gains*Ground_Truth_Sort_Weights)]))
sil_x_lim_min = np.min(np.array([np.min(ESS_silhouette_vals_0),np.min(ESS_silhouette_vals_1)]))

### ESS Plots ###
## Noisy Discretised matrix/No FFAVES Imputation
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(Dropout_Sort_Gains*Dropout_Sort_Weights,cmap="seismic",vmin=imshow_x_lim_min,vmax=1)
ax.set_title("Entropy Sort Scores\nNo FFAVES Adjustment",fontsize=15)
ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_xlabel('Gene IDs',fontsize=13)
plt.xticks(rotation=70)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im,cax=cax)
#
fig, ax = plt.subplots(figsize=(3.5,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Gene_Labels)):
    cluster_silhouette_vals = ESS_silhouette_vals_0[Gene_Labels ==cluster]
    GT_Cluster = GT_ESS_silhouette_vals[Gene_Labels ==cluster]
    GT_Cluster.sort()
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xlim([sil_x_lim_min, 1])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=13)
#ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_title('Gene Clusters Silhouette Plot\nNo FFAVES Adjustment',fontsize=15)
plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

## FFAVES imputed Discretised matrix
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(FFAVES_Imputed_Sort_Gains*FFAVES_Imputed_Sort_Weights,cmap="seismic",vmin=imshow_x_lim_min,vmax=1)
ax.set_title("Entropy Sort Scores\nFFAVES Cycle 8",fontsize=15)
ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_xlabel('Gene IDs',fontsize=13)
plt.xticks(rotation=70)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im,cax=cax)

fig, ax = plt.subplots(figsize=(3.5,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Gene_Labels)):
    cluster_silhouette_vals = ESS_silhouette_vals_1[Gene_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = GT_ESS_silhouette_vals[Gene_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xlim([sil_x_lim_min, 1])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=13)
#ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_title('Gene Clusters Silhouette Plot\nFFAVES Cycle 8',fontsize=15)
plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

## Ground truth discretised matrix
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(Ground_Truth_Sort_Gains*Ground_Truth_Sort_Weights,cmap="seismic",vmin=np.min(Ground_Truth_Sort_Gains*Ground_Truth_Sort_Weights),vmax=1)
ax.set_title("Entropy Sort Scores\nGround Truth",fontsize=15)
ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_xlabel('Gene IDs',fontsize=13)
plt.xticks(rotation=70)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im,cax=cax)

fig, ax = plt.subplots(figsize=(3.5,6))
y_ticks = []
y_lower = y_upper = 0
for i,cluster in enumerate(np.unique(Gene_Labels)):
    cluster_silhouette_vals = GT_ESS_silhouette_vals[Gene_Labels ==cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xlim([sil_x_lim_min, 1])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=13)
#ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_title('Gene Clusters Silhouette Plot\nGround Truth',fontsize=15)
plt.legend(labels=np.array(["1","2","3","4","5","6","7","8"]))
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

### SG + SD plots
## Noisy Discretised matrix/No FFAVES Imputation
imshow_x_lim_min = np.min(np.array([np.min(Dropout_Sort_Gains),np.min(Ground_Truth_Sort_Gains)]))
sil_x_lim_min = np.min(np.array([np.min(IG_silhouette_vals_0),np.min(IG_silhouette_vals_1)]))

fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(Dropout_Sort_Gains,cmap="seismic",vmin=-1,vmax=1)
ax.set_title("Sort Gains and Split Directions\nNo FFAVES Adjustment",fontsize=15)
ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_xlabel('Gene IDs',fontsize=13)
plt.xticks(rotation=70)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im,cax=cax)

fig, ax = plt.subplots(figsize=(3.5,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Gene_Labels)):
    cluster_silhouette_vals = IG_silhouette_vals_0[Gene_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = GT_IG_silhouette_vals[Gene_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xlim([sil_x_lim_min, 1])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=13)
#ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_title('Gene Clusters Silhouette Plot\nNo FFAVES Adjustment',fontsize=15)
plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

## FFAVES imputed Discretised matrix

fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(FFAVES_Imputed_Sort_Gains,cmap="seismic",vmin=-1,vmax=1)
ax.set_title("Sort Gains and Split Directions\nFFAVES Cycle 8",fontsize=15)
ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_xlabel('Gene IDs',fontsize=13)
plt.xticks(rotation=70)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im,cax=cax)

fig, ax = plt.subplots(figsize=(3.5,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Gene_Labels)):
    cluster_silhouette_vals = IG_silhouette_vals_1[Gene_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = GT_IG_silhouette_vals[Gene_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xlim([sil_x_lim_min, 1])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=13)
#ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_title('Gene Clusters Silhouette Plot\nFFAVES Cycle 8',fontsize=15)
plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

## Ground truth discretised matrix
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(Ground_Truth_Sort_Gains,cmap="seismic",vmin=-1,vmax=1)
ax.set_title("Sort Gains and Split Directions\nGround Truth",fontsize=15)
ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_xlabel('Gene IDs',fontsize=13)
plt.xticks(rotation=70)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im,cax=cax)

fig, ax = plt.subplots(figsize=(3.5,6))
y_ticks = []
y_lower = y_upper = 0
for i,cluster in enumerate(np.unique(Gene_Labels)):
    cluster_silhouette_vals = GT_IG_silhouette_vals[Gene_Labels ==cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_xlim([sil_x_lim_min, 1])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=13)
#ax.set_ylabel('Gene IDs',fontsize=13)
ax.set_title('Gene Clusters Silhouette Plot\nGround Truth',fontsize=15)
plt.legend(labels=np.array(["1","2","3","4","5","6","7","8"]))
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()


####### ESFW and Feature Selection Plots #######

Mean_Feature_Divergences_0 = np.load(path+path_2+"Dropout_Feature_Weights.npy")
Mean_Feature_Divergences_1 = np.load(path+path_2+"FFAVES_Imputed_Feature_Weights.npy")
GT_Feature_Divergences_1 = np.load(path+path_2+"Ground_Truth_Feature_Weights.npy")

Max_Value = np.max(np.concatenate((Mean_Feature_Divergences_0,Mean_Feature_Divergences_1,GT_Feature_Divergences_1)))
Gene_IDs = np.arange(Mean_Feature_Divergences_0.shape[0])
Gene_Groups = np.array([[0,49],[50,247],[248,431],[432,612],[613,797],[798,966],[967,1518]])

Imputations = np.zeros(Discretised_Data.shape)
Imputations[Missed_FNs] = 1
Imputations[Identified_FNs] = 2

Errors_Found = np.sum(Imputations==2,axis=0)
Total_Error = np.sum(np.asarray(Imputations!=0),axis=0)
Ratios = Errors_Found/Total_Error
Ratios[np.isnan(Ratios)] = 0

fig, ax = plt.subplots()
scat = ax.scatter(Gene_IDs,Mean_Feature_Divergences_0,c=np.zeros(Ratios.shape[0]),s=8,cmap="magma",vmin=0,vmax=1)
scat2 = ax.scatter(Gene_IDs,GT_Feature_Divergences_1,c="g",s=8,label="Ground truth weights",zorder=-1,edgecolor='g',alpha=0.7)
scat2.set_facecolor('none')
ax.set_ylabel("Feature Weight",fontsize=13)
ax.set_xlabel("Gene ID",fontsize=13)
ax.set_title("Feature Weights\nNo FFAVES Adjustment",fontsize=15)
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylim(0,Max_Value)
cb = plt.colorbar(scat)
cb.set_label(label="Fraction of ground truth\nerror identified by FFAVES", fontsize=14)
plt.legend()

fig, ax = plt.subplots()
scat = ax.scatter(Gene_IDs,Mean_Feature_Divergences_1,c=Ratios,s=8,cmap="magma",vmin=0,vmax=1)
scat2 = ax.scatter(Gene_IDs,GT_Feature_Divergences_1,c="g",s=8,label="Ground truth weights",zorder=-1,edgecolor='g',alpha=0.7)
scat2.set_facecolor('none')
ax.set_ylabel("Feature Weight",fontsize=13)
ax.set_xlabel("Gene ID",fontsize=13)
ax.set_title("Feature Weights\nFFAVES Cycle 8",fontsize=15)
ax.set_xticks(np.append(Gene_Groups[:,0],Gene_Groups[-1,-1]))
ax.set_ylim(0,Max_Value)
cb = plt.colorbar(scat)
cb.set_label(label="Fraction of ground truth\nerror identified by FFAVES", fontsize=14)
plt.legend()

plt.figure(figsize=(6,5))
plt.subplot(2, 1, 1)
plt.hist(GT_Feature_Divergences_1,bins=30,color="g",alpha=0.7,label="Ground truth weights")
plt.hist(Mean_Feature_Divergences_0,bins=30,label="No FFAVES adjustment weights")
plt.ylabel("Frequency",fontsize=12)
#plt.xlabel("Feature Weight",fontsize=13)
plt.xlim(0,Max_Value)
plt.title("Feature weights before and after FFAVES adjustment",fontsize=14)
plt.legend()

plt.subplot(2, 1, 2)
plt.hist(GT_Feature_Divergences_1,bins=30,color="g",alpha=0.7,label="Ground truth weights")
plt.hist(Mean_Feature_Divergences_1,bins=30,label="FFAVES cycle 8 weights")
plt.ylabel("Frequency",fontsize=12)
plt.xlabel("Feature Weight",fontsize=12)
plt.xlim(0,Max_Value)
plt.legend()
plt.tight_layout()

plt.show()

### Plot Highly Variable Gene Precision-Recall Curves ###
# The ESFW and ESFW + FFAVES gene rankings are simply the ranked feature weights obtained by the ESFW function.
# Higher scores are better.
# ESFW scores with no FFAVES imputation. 
FFAVES_HVG_Order_0 = np.argsort(-Mean_Feature_Divergences_0)
# ESFW scores with converged FFAVES imputation.
FFAVES_HVG_Order_1 = np.argsort(-Mean_Feature_Divergences_1)
# Load ranked gene list from SCRAN, Seurat and scry. See the "Get_Information_Rich_Gene_Rankings.r" 
# file for instructions on creating the following objects.
SCRAN_HVG_Order = pd.read_csv(path+path_2+"SCRAN_HVG_Order.csv",header=0,index_col=0)
SCRAN_HVG_Order = np.asarray(SCRAN_HVG_Order.T)[0]
Seurat_HVG_Order = pd.read_csv(path+path_2+"Seurat_HVG_Order.csv",header=0,index_col=0)
Seurat_HVG_Order = np.asarray(Seurat_HVG_Order.T)[0]
scry_HVG_Order = pd.read_csv(path+path_2+"scry_HVG_Order.csv",header=0,index_col=0)
scry_HVG_Order = np.asarray(scry_HVG_Order.T)[0]

Structured_Genes = np.arange(970)
Random_Genes = np.arange(970,1519)
Top_Percentages = np.linspace(1,100,100)/100
FFAVES_Precisions_0 = np.zeros(Top_Percentages.shape[0])
FFAVES_Precisions_1 = np.zeros(Top_Percentages.shape[0])
SCRAN_Precisions = np.zeros(Top_Percentages.shape[0])
Seurat_Precisions = np.zeros(Top_Percentages.shape[0])
scry_Precisions = np.zeros(Top_Percentages.shape[0])
FFAVES_Recall_0 = np.zeros(Top_Percentages.shape[0])
FFAVES_Recall_1 = np.zeros(Top_Percentages.shape[0])
SCRAN_Recall = np.zeros(Top_Percentages.shape[0])
Seurat_Recall = np.zeros(Top_Percentages.shape[0])
scry_Recall = np.zeros(Top_Percentages.shape[0])

for i in np.arange(Top_Percentages.shape[0]):
    Top_Gene_Range = int(FFAVES_HVG_Order_0.shape[0] * Top_Percentages[i])
    FFAVES_Genes = FFAVES_HVG_Order_0[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,FFAVES_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,FFAVES_Genes))
    FFAVES_Precisions_0[i] = Num_Structured/Structured_Genes.shape[0]  
    FFAVES_Recall_0[i] = Num_Structured/Top_Gene_Range
    #
    Top_Gene_Range = int(FFAVES_HVG_Order_1.shape[0] * Top_Percentages[i])
    FFAVES_Genes = FFAVES_HVG_Order_1[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,FFAVES_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,FFAVES_Genes))
    FFAVES_Precisions_1[i] = Num_Structured/Structured_Genes.shape[0]  
    FFAVES_Recall_1[i] = Num_Structured/Top_Gene_Range
    #
    SCRAN_Genes = SCRAN_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,SCRAN_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,SCRAN_Genes))
    SCRAN_Precisions[i] = Num_Structured/Structured_Genes.shape[0]  
    SCRAN_Recall[i] = Num_Structured/Top_Gene_Range
    #
    Seurat_Genes = Seurat_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,Seurat_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,Seurat_Genes))
    Seurat_Precisions[i] = Num_Structured/Structured_Genes.shape[0]  
    Seurat_Recall[i] = Num_Structured/Top_Gene_Range
    #
    scry_Genes = scry_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,scry_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,scry_Genes))
    scry_Precisions[i] = Num_Structured/Structured_Genes.shape[0]  
    scry_Recall[i] = Num_Structured/Top_Gene_Range

plt.figure()
plt.plot(FFAVES_Precisions_0,FFAVES_Recall_0,label="ESFW without FFAVES")
plt.plot(FFAVES_Precisions_1,FFAVES_Recall_1,label="ESFW + FFAVES after 7 cycles",linestyle=(0, (5, 10)),linewidth=3)
plt.plot(SCRAN_Precisions,SCRAN_Recall,label="Scran")
plt.plot(Seurat_Precisions,Seurat_Recall,label="Seurat",c='#9467bd')
plt.plot(scry_Precisions,scry_Recall,label="scry",c="#e377c2",alpha=0.7)
plt.axhline(Structured_Genes.shape[0]/Mean_Feature_Divergences_0.shape[0],linestyle="--",c="r",zorder=-1,label="Equivalent to random sampling")
plt.title("Feature Selection\nPrecision-Recall",fontsize=15)
plt.xlabel("Recall",fontsize=13)
plt.ylabel("Precision",fontsize=13)
plt.legend(facecolor='white', framealpha=1)

plt.show()


### Plot UMAPs for imputed data with different methods ###
# Load starting data
Complete_Synthetic_Data = pd.read_csv(path+path_2+"Complete_Synthetic_Data.csv",header=0,index_col=0)
Drop_Out_Synthetic_Data = pd.read_csv(path+path_2+"Drop_Out_Synthetic_Data.csv",header=0,index_col=0)
# Load each imputed data. See the "Get_Imputed_Expression_Matricies.r" file for the creation of these objects.
Synthetic_Imputation_MAGIC = pd.read_csv(path+path_2+"Synthetic_Imputation_MAGIC_Rounded.csv",header=0,index_col=0)
Synthetic_Imputation_SAVER = pd.read_csv(path+path_2+"Synthetic_Imputation_SAVER.csv",header=0,index_col=0).T
Synthetic_Imputation_FFAVES = pd.read_csv(path+path_2+"Synthetic_Imputation_FFAVES.csv",header=0,index_col=0)
Synthetic_Imputation_ALRA = pd.read_csv(path+path_2+"Synthetic_Imputation_ALRA.csv",header=0,index_col=0).T

Batch_1 = np.arange(0,Synthetic_Imputation_MAGIC.shape[0],2)
Batch_2 = np.arange(1,Synthetic_Imputation_MAGIC.shape[0],2)

Neighbours = 20
Dist = 0.1
Complete_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Complete_Synthetic_Data)
Noisy_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Drop_Out_Synthetic_Data)
MAGIC_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_MAGIC)
SAVER_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_SAVER)
FFAVES_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_FFAVES)
ALRA_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_ALRA)

Cell_Labels = np.zeros(200)
for i in np.arange(1,6):
    Cell_Labels = np.append(Cell_Labels,np.repeat(i,200))

Unique_Cell_Labels = np.unique(Cell_Labels)
size = 0.5
# create 3x1 subplots
fig, axs = plt.subplots(nrows=6, ncols=1,figsize = (5,8))
# clear subplots
for ax in axs:
    ax.remove()

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]
Row_Titles = np.array(["Ground Truth","30% Dropouts + Batch Effects","FFAVES Imputation","MAGIC Imputation","ALRA Imputation","SAVER Imputation"])
Row_Embeddings = [Complete_Embedding,Noisy_Embedding,FFAVES_Embedding,MAGIC_Embedding,ALRA_Embedding,SAVER_Embedding]

for row, subfig in enumerate(subfigs):
    subfig.suptitle(Row_Titles[row],fontsize=10)
    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=2)
    for col, ax in enumerate(axs):
        if col == 0:
           for i in np.arange(Unique_Cell_Labels.shape[0]):
                IDs = np.where(Cell_Labels == Unique_Cell_Labels[i])[0]
                if Unique_Cell_Labels[i] == 5:
                    ax.scatter(Row_Embeddings[row][IDs,0],Row_Embeddings[row][IDs,1],s=size,zorder=-1)
                else:
                    ax.scatter(Row_Embeddings[row][IDs,0],Row_Embeddings[row][IDs,1],s=size) 
        if col == 1:
            ax.scatter(Row_Embeddings[row][Batch_1,0],Row_Embeddings[row][Batch_1,1],c="black",alpha=0.8,s=size)
            ax.scatter(Row_Embeddings[row][Batch_2,0],Row_Embeddings[row][Batch_2,1],c="gold",alpha=0.4,s=size)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

plt.show()

### Plot silhoutte scores for each synthetic cell type to quantify how well they cluser together before and after imputation ###


Cell_Labels = np.zeros(200)
for i in np.arange(1,6):
    Cell_Labels = np.append(Cell_Labels,np.repeat(i,200))

Complete_silhouette_vals = silhouette_samples(Complete_Synthetic_Data,Cell_Labels)
Drop_Out_silhouette_vals = silhouette_samples(Drop_Out_Synthetic_Data,Cell_Labels)
MAGIC_silhouette_vals = silhouette_samples(Synthetic_Imputation_MAGIC,Cell_Labels)
SAVER_silhouette_vals = silhouette_samples(Synthetic_Imputation_SAVER,Cell_Labels)
FFAVES_silhouette_vals = silhouette_samples(Synthetic_Imputation_FFAVES,Cell_Labels)
ALRA_silhouette_vals = silhouette_samples(Synthetic_Imputation_ALRA,Cell_Labels)

Min_Silh = np.min(np.concatenate([Complete_silhouette_vals,Drop_Out_silhouette_vals,MAGIC_silhouette_vals,SAVER_silhouette_vals,FFAVES_silhouette_vals]))
Max_Silh = np.max(np.concatenate([Complete_silhouette_vals,Drop_Out_silhouette_vals,MAGIC_silhouette_vals,SAVER_silhouette_vals,FFAVES_silhouette_vals]))

fig, ax = plt.subplots(figsize=(4,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = Complete_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.arange(0,1200,200))
ax.set_xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15)
ax.set_title('Ground Truth',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()

fig, ax = plt.subplots(figsize=(4,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = Drop_Out_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.arange(0,1200,200))
ax.set_xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('30% Dropouts + Batch Effects',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()

fig, ax = plt.subplots(figsize=(4,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = FFAVES_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.arange(0,1200,200))
ax.set_xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('FFAVES',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()

fig, ax = plt.subplots(figsize=(4,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = MAGIC_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.arange(0,1200,200))
ax.set_xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15)
ax.set_title('MAGIC',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()

fig, ax = plt.subplots(figsize=(4,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = SAVER_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.arange(0,1200,200))
ax.set_xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('SAVER',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.tight_layout()
plt.gca().invert_yaxis()

fig, ax = plt.subplots(figsize=(4,6))
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = ALRA_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      


ax.set_yticks(np.arange(0,1200,200))
ax.set_xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('ALRA',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),fontsize=15)
plt.tight_layout()
plt.gca().invert_yaxis()














