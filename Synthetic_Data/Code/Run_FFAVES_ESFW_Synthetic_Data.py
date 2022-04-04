# The following code takes the synthetic data presented in the main paper and uses FFAVES and ESFW
# to identify spurious data points according to the prevailing structure in the data.

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
import ffaves

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

### Run FFAVES and ESFW ###
## Auto_Save set to False so not to overwrite objects used for paper or save in undersired directory.
## Discretised_Data should be a numpy array of 0's and 1's with the rows as samples and the columns as features.
# FFAVES
Track_Imputation_Steps, Track_Percentage_Imputation = ffaves.FFAVES(Discretised_Data, Auto_Save=False)
# Calculate correlation matricies
Chosen_Cycle = -1
Sort_Gains, Sort_Weights, Cycle_Suggested_Imputations, ES_Matrices_Features_Used_Inds = ffaves.Calculate_ES_Sort_Matricies(Discretised_Data, Track_Imputation_Steps, Chosen_Cycle=Chosen_Cycle, Auto_Save=False)
# ESFW
Feature_Divergences, Cycle_Suggested_Imputations, Feature_Divergences_Used_Inds = ffaves.ESFW(0.1, Discretised_Data, Track_Imputation_Steps, Chosen_Cycle=Chosen_Cycle, Auto_Save=False)
# Feature weights
Mean_Feature_Divergences = np.mean(Feature_Divergences,axis=0)
# Identify optimised discretisation thresholds based on the discretised matrix before and after application of FFAVES. 
# This is a cursory function for demonstration of FFAVES's ability to account for sub-optimal discretisation. It is an optional step in the
# suggested FFAVES workflow and was primarily created for synthetic data with known ground truth rather than complex real data.
Optimised_Thresholds = ffaves.Parallel_Optimise_Discretisation_Thresholds(np.asarray(Drop_Out_Synthetic_Data).astype("f"),Discretised_Data.astype("f"),Cycle_Suggested_Imputations, Auto_Save=False)

# Typically the function paramter Auto_Save defaults to True, meaning files are automatically saved to the current directory.
# However, to keep the directories tidy for this our synthetica data example, we will save to the data directory pulled from github.
np.save(path+path_2+"Track_Percentage_Imputation.npy",Track_Percentage_Imputation)
np.save(path+path_2+"Track_Imputation_Steps.npy",Track_Imputation_Steps)
np.save(path+path_2+"Sort_Gains.npy",Sort_Gains)
np.save(path+path_2+"Sort_Weights.npy",Sort_Weights)
np.save(path+path_2+"Cycle_Suggested_Imputations.npy",Cycle_Suggested_Imputations)
np.save(path+path_2+"Feature_Divergences.npy",Feature_Divergences)
np.save(path+path_2+"ES_Matrices_Features_Used_Inds.npy",ES_Matrices_Features_Used_Inds)
np.save(path+path_2+"Optimised_Thresholds.npy",Optimised_Thresholds)

### Load objects for basic plotting ###
# Logs the how many points in the Discretised_Data have been suggested as false negative (FN) and false positive (FP) data points after each itteration of FFAVES.
Track_Percentage_Imputation = np.load(path+path_2+"Track_Percentage_Imputation.npy")
# The Sort Gain (SG) and Sort Weight (SW) scores for each pair of features in the data, for the given cycle of FFAVES that
# was used for suggesting FN/FP data points (Chosen_Cycle = -1 indicates the last cycle of FFAVES). Technically the Sort_Gains matrix
# is a product of the Sort Gain and Sort Direction scores, hence values can range from -1 to 1. This was done to save memory. To obtain the true
# Sort Gain and Sort Direction scores, extract all negative values as -1 and all positive values as 1 and then get 
# the absolute values of the Sort_Gains matrix.
Sort_Gains = np.load(path+path_2+"Sort_Gains.npy")
Sort_Weights = np.load(path+path_2+"Sort_Weights.npy")
# IDs all of the feature columns that were used. Indicies that are not present are features that were exlcuded because the number of minority states
# observed in the feature was less than Min_Clust_Size (default = 5).
ES_Matrices_Features_Used_Inds = np.load(path+path_2+"ES_Matrices_Features_Used_Inds.npy")

### Plot basic plots ###
# Plot the convergence of suggested FN/FP data points
plt.figure()
plt.plot(Track_Percentage_Imputation[0,:],label="False Positives")
plt.plot(Track_Percentage_Imputation[1,:],label="False Negatives")
plt.title("Synthetic Data Set \n Suggested FPs/FNs after each cycle of FFAVES",fontsize=16)
plt.xlabel("FFAVES Cycle",fontsize=13)
plt.ylabel("Proportion of M identified as \n FP or FN (%)",fontsize=13)
plt.xticks(np.arange(0, 10))
plt.legend()
# Plot pairwise feature SGs * SDs
plt.figure()
plt.imshow(Sort_Gains,cmap="seismic")
plt.title("Sort Gain and Split Direction",fontsize=16)
plt.xlabel("Gene IDs")
plt.ylabel("Gene IDs")
plt.colorbar()
# Plot pairwise feature ESSs
plt.figure()
plt.imshow(Sort_Gains*Sort_Weights,cmap="seismic",vmax=1,vmin=-1)
plt.title("Entropy Sort Scores",fontsize=16)
plt.xlabel("Gene IDs")
plt.ylabel("Gene IDs")
plt.colorbar()

Feature_Divergences = np.load(path+path_2+"Feature_Divergences.npy")
Mean_Feature_Divergences = np.mean(Feature_Divergences,axis=0)
# Inspect estimated feature weights
plt.figure()
plt.title("ESFW Estimated Feature Weights",fontsize=16)
plt.hist(Mean_Feature_Divergences,bins=30)
plt.xlabel("Feature Weights",fontsize=14)
plt.ylabel("Frequency",fontsize=14)

plt.show()

### Having applied FFAVES and ESFW to the synthetic data, proceed to the Synthetic_Data_Plotting file to plot more detailed figures.

