# The following code takes the synthetic data from the paper and the SAVER, MAGIC and ALRA
# software to attempt to impute the dropouts. These first 3 software are R software.
# The final software usage is the application of the python package IterativeImputer, to 
# estimate the FN data points suggested by FFAVES after convergence is acheived.

### Set folder path as required ###
path = "/mnt/c/Users/arthu/OneDrive - University of Cambridge/Entropy_Sorting_Paper_2022/"
path_2 = "Synthetic_Data/Objects_For_Paper/"
###

##### R Code Imputation of Synthetic Data ######
counts <- read.csv("Drop_Out_Synthetic_Data.csv")
counts <- counts[,-1]
# Columns are cells, rows are genes
counts <- t(counts)

### drImpute ###
library(DrImpute)

X.imp <- DrImpute(counts,mc.cores = 10)
## Un-comment to save results ##
#write.csv(X.imp,"Synthetic_Imputation_drImpute.csv")

### SAVER ###
library(SAVER)

counts.saver <- saver(counts, ncores = 10)
## Un-comment to save results ##
#write.csv(counts.saver$estimate,"Synthetic_Imputation_SAVER.csv")

### MAGIC ###
library(Rmagic)

MAGIC_data <- magic(t(counts))
## Un-comment to save results ##
#write.csv(MAGIC_data$result,"Synthetic_Imputation_MAGIC.csv")

#### ALRA ####
library(SeuratWrappers)

s_obj <- RunALRA(s_obj)
ALRA <- as.data.frame(s_obj@assays$alra@data)
## Un-comment to save results ##
#write.csv(ALRA,"Synthetic_Imputation_ALRA.csv")


##### Python Code Imputation of Synthetic Data ######

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np
import copy

Drop_Out_Synthetic_Data = pd.read_csv(path+path_2+"Drop_Out_Synthetic_Data.csv",header=0,index_col=0)
Track_Imputation_Steps = np.load(path+path_2+"Track_Imputation_Steps.npy",allow_pickle=True)
# Use final cycle of FFAVES suggested imputations (reached convergence)
Chosen_Cycle = -1

imputer = IterativeImputer(BayesianRidge(),n_nearest_features=200,max_iter=10,sample_posterior=True,initial_strategy="median",min_value=0)

Impute_Data = copy.copy(np.asarray(Drop_Out_Synthetic_Data))
# Set suggested FNs to nan values for imputer to estimate.
Impute_Data[Track_Imputation_Steps[Chosen_Cycle][2]] = np.nan

Imputed_Data = imputer.fit_transform(Impute_Data)
FFAVES_Imputed_Data = pd.DataFrame(Imputed_Data)
## Un-comment to save results ##
#FFAVES_Imputed_Data.to_csv("Synthetic_Imputation_FFAVES.csv")

